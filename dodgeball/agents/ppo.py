"""Proximal Policy Optimisation with a shared actor-critic network (self-play)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dodgeball.agents.opponents import OpponentPool


class RolloutBuffer:
    """One rollout stored as (time, agent) arrays, so each agent's own trajectory can be
    followed when computing advantages. ``valid`` marks the entries where the agent acted."""

    def __init__(self, steps, n_agents, obs_size, n_actions):
        self.obs = torch.zeros((steps, n_agents, obs_size))
        self.actions = torch.zeros((steps, n_agents), dtype=torch.long)
        self.action_masks = torch.zeros((steps, n_agents, n_actions))
        self.logprobs = torch.zeros((steps, n_agents))
        self.rewards = torch.zeros((steps, n_agents))
        self.values = torch.zeros((steps, n_agents))
        self.dones = torch.ones((steps, n_agents))
        self.valid = torch.zeros((steps, n_agents), dtype=torch.bool)

    def clear(self):
        self.rewards.zero_()
        self.values.zero_()
        self.dones.fill_(1.0)   # entries where the agent didn't act count as "done"
        self.valid.fill_(False)


class PPO(nn.Module):

    def __init__(self, n_actions, obs_size):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs):
        shared = self.backbone(obs)
        logits = self.policy_head(shared)
        values = self.value_head(shared).squeeze(-1)
        return logits, values

    @staticmethod
    def masked_dist(logits, action_mask):
        mask = torch.as_tensor(action_mask, dtype=torch.float32)
        return torch.distributions.Categorical(logits=logits + torch.log(mask + 1e-8))

    def select_action_and_value(self, obs, action_mask):
        logits, value = self.forward(obs)
        dist = self.masked_dist(logits, action_mask)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class PPOTrainer:

    TIME = 2048          # environment steps per update
    GAMMA = 0.99         # discount factor
    GAE = 0.95           # GAE lambda
    CLIP = 0.2           # PPO clip range
    EPOCHS = 4           # optimisation epochs per update
    BATCH_SIZE = 64
    LR = 3e-4
    ENT_COEF = 0.01      # entropy bonus (computed on the masked distribution)
    VF_COEF = 0.5        # value loss weight
    NORM_ADV = False     # normalise advantages per update (--hp NORM_ADV=True)

    # Opponent mix (see opponents.py). Defaults are pure self-play.
    P_SELF = 1.0         # chance an episode is played against the current network
    P_HEURISTIC = 0.0    # ... against the scripted heuristic
    P_PAST = 0.0         # ... against a frozen past copy of the network
    P_RANDOM = 0.0       # ... against the random policy
    SNAPSHOT_EVERY = 10  # updates between snapshots added to the past-opponent pool
    POOL_SIZE = 5        # snapshots kept (oldest dropped first)

    def __init__(self, env):
        self.env = env
        n_actions = env.n_actions
        obs_size = env.observation_space(0).shape[0]
        self.n_agents = len(env.possible_agents)
        self.policy_net = PPO(n_actions, obs_size)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.buffer = RolloutBuffer(self.TIME, self.n_agents, obs_size, n_actions)
        self.env_steps = 0   # environment steps so far
        self.opponents = OpponentPool(env, "ppo", self.P_SELF, self.P_HEURISTIC, self.P_PAST, self.POOL_SIZE,
                                      self.P_RANDOM)

    def optimize_model(self, loader):
        for _ in range(self.EPOCHS):
            for batch_obs, batch_masks, batch_actions, batch_logprob, batch_returns, batch_advantages in loader:
                logits, values = self.policy_net(batch_obs)
                dist = self.policy_net.masked_dist(logits, batch_masks)
                new_logprob = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprob - batch_logprob)
                clipped = torch.clamp(ratio, 1 - self.CLIP, 1 + self.CLIP) * batch_advantages
                policy_loss = -torch.min(ratio * batch_advantages, clipped).mean()
                value_loss = F.mse_loss(values, batch_returns)
                loss = policy_loss + self.VF_COEF * value_loss - self.ENT_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_advantages(self, last_values):
        """GAE, computed separately for each agent along its own trajectory."""
        buf = self.buffer
        advantages = torch.zeros_like(buf.rewards)
        next_value = last_values
        next_advantage = torch.zeros(self.n_agents)
        for t in reversed(range(self.TIME)):
            not_done = 1.0 - buf.dones[t]
            delta = buf.rewards[t] + self.GAMMA * next_value * not_done - buf.values[t]
            advantages[t] = delta + self.GAMMA * self.GAE * not_done * next_advantage
            next_value = buf.values[t]
            next_advantage = advantages[t]
        return advantages, advantages + buf.values

    def train(self, num_updates, callback=None):
        """Run ``num_updates`` PPO updates and return the total reward collected in each.

        ``callback(update)`` is called after every update (used for periodic evaluation).
        """
        buf = self.buffer
        pool = self.opponents
        obs, infos = self.env.reset()
        pool.sample_episode()
        update_rewards = []
        last_step = {}   # agent -> rollout index of its latest step in the current episode

        for i_update in range(1, num_updates + 1):
            buf.clear()
            rewards_total = 0
            last_step.clear()   # earlier steps were already used for training

            # Collect a rollout.
            for t in range(self.TIME):
                actions = {}
                for agent in self.env.agents:
                    mask = infos[agent]["action_mask"]
                    if not pool.is_learner(agent):
                        # Opponent agents act but aren't stored: buf.valid stays False and
                        # buf.dones stays 1, so GAE never chains through them.
                        actions[agent] = pool.opponent.act(obs[agent], mask)
                        continue
                    agent_obs = torch.as_tensor(obs[agent], dtype=torch.float32)
                    with torch.no_grad():
                        action, logprob, _, value = self.policy_net.select_action_and_value(agent_obs, mask)
                    buf.obs[t, agent] = agent_obs
                    buf.action_masks[t, agent] = torch.as_tensor(mask, dtype=torch.float32)
                    buf.actions[t, agent] = action
                    buf.logprobs[t, agent] = logprob
                    buf.values[t, agent] = value
                    buf.valid[t, agent] = True
                    last_step[agent] = t
                    actions[agent] = action.item()

                obs, rewards, terminated, truncated, infos = self.env.step(actions)
                self.env_steps += 1
                learners = [a for a in actions if pool.is_learner(a)]
                rewards_total += sum(rewards[a] for a in learners)

                for agent in learners:
                    buf.rewards[t, agent] = rewards[agent]
                    buf.dones[t, agent] = float(terminated[agent] or truncated[agent])

                # Win/lose reward for players eliminated earlier (RewardConfig.terminal_to_eliminated)
                for agent, reward in self.env.late_rewards.items():
                    if agent in last_step:
                        buf.rewards[last_step[agent], agent] += reward
                        rewards_total += reward

                if not self.env.agents:
                    obs, infos = self.env.reset()
                    pool.sample_episode()
                    last_step.clear()

            update_rewards.append(rewards_total)
            print(f"update {i_update}/{num_updates}  total reward: {rewards_total:.1f}")

            # Bootstrap from the value of agents still mid-game at the end of the rollout.
            last_values = torch.zeros(self.n_agents)
            with torch.no_grad():
                for agent in self.env.agents:
                    if not pool.is_learner(agent):
                        continue
                    _, last_values[agent] = self.policy_net(torch.as_tensor(obs[agent], dtype=torch.float32))

            advantages, returns = self.compute_advantages(last_values)

            valid = buf.valid
            if self.NORM_ADV:
                adv = advantages[valid]
                advantages[valid] = (adv - adv.mean()) / (adv.std() + 1e-8)
            dataset = torch.utils.data.TensorDataset(
                buf.obs[valid], buf.action_masks[valid], buf.actions[valid],
                buf.logprobs[valid], returns[valid], advantages[valid],
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)
            self.optimize_model(loader)
            if i_update % self.SNAPSHOT_EVERY == 0:
                pool.snapshot(self.policy_net)
            if callback is not None:
                callback(i_update)

        return update_rewards

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
