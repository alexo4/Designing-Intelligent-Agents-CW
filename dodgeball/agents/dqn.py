"""Deep Q-Network with a single network shared by every player (self-play)."""

import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dodgeball.agents.opponents import OpponentPool

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "next_action_mask"))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.pushed = 0   # total transitions ever pushed; used to find old ones by id

    def push(self, *args):
        """Store a transition and return its id (for ``add_reward``)."""
        self.memory.append(Transition(*args))
        self.pushed += 1
        return self.pushed - 1

    def add_reward(self, transition_id, reward):
        """Add ``reward`` to an earlier transition, if it is still in memory."""
        pos = transition_id - (self.pushed - len(self.memory))
        if pos >= 0:
            t = self.memory[pos]
            self.memory[pos] = t._replace(reward=t.reward + reward)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_actions, obs_size):
        super().__init__()
        self.fc_combined = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, obs):
        return self.fc_combined(obs.float())


class DQNTrainer:

    BATCH_SIZE = 128     # transitions sampled from the replay buffer
    GAMMA = 0.99         # discount factor
    EPS_START = 0.9      # starting epsilon
    EPS_END = 0.05       # final epsilon
    EPS_DECAY = 800000     # epsilon decay rate (higher = slower)
    TAU = 0.005          # soft update rate of the target network
    LR = 1e-4            # AdamW learning rate
    MEMORY_SIZE = 5000000
    DOUBLE = False       # use the Double DQN target (--hp DOUBLE=True)

    # Opponent mix (see opponents.py). Defaults are pure self-play.
    P_SELF = 1.0         # chance an episode is played against the current network
    P_HEURISTIC = 0.0    # ... against the scripted heuristic
    P_PAST = 0.0         # ... against a frozen past copy of the network
    P_RANDOM = 0.0       # ... against the random policy
    SNAPSHOT_EVERY = 500 # episodes between snapshots added to the past-opponent pool
    POOL_SIZE = 5        # snapshots kept (oldest dropped first)

    def __init__(self, env):
        self.env = env
        n_actions = env.n_actions
        obs_size = env.observation_space(0).shape[0]
        self.policy_net = DQN(n_actions, obs_size)
        self.target_net = DQN(n_actions, obs_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(self.MEMORY_SIZE)
        self.steps_done = 0  # action selections so far; drives the epsilon decay
        self.env_steps = 0   # environment steps so far
        self.opponents = OpponentPool(env, "dqn", self.P_SELF, self.P_HEURISTIC, self.P_PAST, self.POOL_SIZE,
                                      self.P_RANDOM)

    def select_action(self, obs, action_mask):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(obs)
                masked_q = q_values + torch.log(torch.from_numpy(action_mask).float() + 1e-8)
                return torch.argmax(masked_q).item()
        valid_actions = np.where(action_mask == 1)[0]
        return np.random.choice(valid_actions)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
        non_final_next_states = torch.stack([
            torch.FloatTensor(np.array([s])) for s in batch.next_state if s is not None
        ])
        state_batch = torch.cat([torch.Tensor(np.array([s])) for s in batch.state])
        action_batch = torch.cat([torch.LongTensor([a]) for a in batch.action])
        reward_batch = torch.cat([torch.FloatTensor([r]) for r in batch.reward])

        # Q(s_t, a) for the actions that were taken
        q_values = self.policy_net(state_batch).squeeze(1)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1))

        # max over valid actions of Q_target(s_{t+1}, a), or 0 for terminal states.
        # Invalid actions are never taken, so their Q-values are never trained and must be
        # excluded from the max, otherwise the targets drift upwards without bound.
        next_masks = torch.as_tensor(np.array([m for s, m in zip(batch.next_state, batch.next_action_mask)
                                               if s is not None]))
        next_state_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            next_q_values = self.target_net(non_final_next_states).reshape(len(next_masks), -1)
            if self.DOUBLE:
                # Double DQN: the policy net picks the action, the target net scores it,
                # which reduces the overestimation bias of taking a max over noisy values.
                policy_q = self.policy_net(non_final_next_states).reshape(len(next_masks), -1)
                policy_q[next_masks == 0] = -float("inf")
                best_actions = policy_q.argmax(1, keepdim=True)
                next_state_values[non_final_mask] = next_q_values.gather(1, best_actions).squeeze(1)
            else:
                next_q_values[next_masks == 0] = -float("inf")
                next_state_values[non_final_mask] = next_q_values.max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def soft_update_target(self):
        """theta' <- tau * theta + (1 - tau) * theta'"""
        target_state = self.target_net.state_dict()
        policy_state = self.policy_net.state_dict()
        for key in policy_state:
            target_state[key] = policy_state[key] * self.TAU + target_state[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_state)

    def train(self, num_episodes, log_every=10, callback=None, max_env_steps=None):
        """Train for ``num_episodes`` and return the total reward of each episode.

        With ``max_env_steps``, training also stops after the first episode that reaches that
        many environment steps (so DQN and PPO can be given the same budget).
        ``callback(episode)`` is called after every episode (used for periodic evaluation).
        """
        episode_rewards = []

        pool = self.opponents

        for i_episode in range(num_episodes):
            obs, infos = self.env.reset()
            pool.sample_episode()
            rewards_total = 0
            last_transition = {}   # agent -> id of its latest transition this episode

            while self.env.agents:
                actions = {}
                for agent in self.env.agents:
                    mask = infos[agent]["action_mask"]
                    if pool.is_learner(agent):
                        agent_obs = torch.tensor(obs[agent], dtype=torch.float32)
                        actions[agent] = self.select_action(agent_obs, mask)
                    else:
                        actions[agent] = pool.opponent.act(obs[agent], mask)

                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                self.env_steps += 1

                # Only the network being trained learns from its own experience.
                learners = [a for a in actions if pool.is_learner(a)]
                rewards_total += sum(rewards[a] for a in learners)
                for agent in learners:
                    last_transition[agent] = self.memory.push(
                        [obs[agent]],
                        actions[agent],
                        None if terminated[agent] else [next_obs[agent]],
                        rewards[agent],
                        infos[agent]["action_mask"],
                    )
                # Win/lose reward for players eliminated earlier (RewardConfig.terminal_to_eliminated)
                for agent, reward in self.env.late_rewards.items():
                    if agent in last_transition:
                        self.memory.add_reward(last_transition[agent], reward)
                        rewards_total += reward

                obs = next_obs
                self.optimize_model()
                self.soft_update_target()

            episode_rewards.append(rewards_total)
            if (i_episode + 1) % self.SNAPSHOT_EVERY == 0:
                pool.snapshot(self.policy_net)
            if (i_episode + 1) % log_every == 0:
                recent = np.mean(episode_rewards[-log_every:])
                print(f"episode {i_episode + 1}/{num_episodes}  mean reward (last {log_every}): {recent:.1f}")
            if callback is not None:
                callback(i_episode + 1)
            if max_env_steps is not None and self.env_steps >= max_env_steps:
                print(f"reached {self.env_steps} environment steps after {i_episode + 1} episodes")
                break

        return episode_rewards

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
