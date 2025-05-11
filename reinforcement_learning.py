import math
import random
from collections import namedtuple, deque

import numpy as np
import torch.optim as optim
from torch import device

from environment import Environment
import torch
import torch.nn as nn


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_global = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )

        self.fc_team = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU()
        )

        grid_size = 4096
        global_info = 16
        team_info = 32
        total_input = grid_size + global_info + team_info
        self.fc_combined = nn.Sequential(
            nn.Linear(total_input, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, obs):

        grid_features = self.conv(obs["observations"]["court"])
        global_features = self.fc_global(obs["observations"]["global_info"])
        team_features = self.fc_team(
            torch.cat([
                obs["observations"]["team_info"][0],
                obs["observations"]["team_info"][1]
            ], dim=1)
        )
        combined = torch.cat([grid_features, global_features, team_features])
        q_values = self.fc_combined(combined)
        masked_q = q_values + torch.log(obs["action_mask"].float())
        return masked_q


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = Environment(8, 12, 5)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_actions)
target_net = DQN(n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(obs):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            q_values = policy_net(obs)
            return torch.argmax(q_values).item()
    else:
        valid_actions = np.where(obs["action_mask"] == 1)[0]
        return np.random.choice(valid_actions)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) # can also use nn.MSELoss

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 50

for i_episode in range(num_episodes):

    obs = env.reset()
    done = False

    while not done:

        actions = {}

        for agent in env.agents:
            agent_obs = {
                "observations": torch.FloatTensor(obs[agent]["observations"]),
                "action_mask": torch.FloatTensor(obs[agent]["action_mask"])
            }
            actions[agent] = select_action(agent_obs)

        next_obs, rewards, terminated, truncated, info = env.step(actions)

        for agent in env.agents:
            memory.push(obs[agent]["observations"], actions[agent], next_obs[agent]["observations"], rewards[agent])

        for agent in env.agents:
            if terminated[agent]:
                env.agents.remove(agent)

        obs = next_obs

        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if all(list(truncated.values())):
            done = True










