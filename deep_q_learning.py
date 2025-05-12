import math
import pickle
import queue
import random
import threading
import time
from collections import namedtuple, deque

import numpy as np
import torch.optim as optim
from gym.spaces import unflatten

from window import Window
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

        grid_size = 768
        global_info = 4
        team_info = 12
        total_input = grid_size + global_info + team_info
        self.fc_combined = nn.Sequential(
            nn.Linear(total_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)

        )

    def forward(self, obs):
        torch.set_printoptions(threshold=10_000)


        #court = obs["court"].float()
        #print(court.shape)
        #court = court.permute(2, 0, 1).unsqueeze(0)
        #grid_features = self.conv(court).squeeze(0)
       # global_features = self.fc_global(obs["global_info"])
        #team_features = self.fc_team(obs["team_info"])
        #print(grid_features.shape)
        #print(global_features.shape)
        #print(team_features.shape)
       # combined = torch.cat([grid_features, global_features, team_features])
       # print(combined.shape)
      #  q_values = self.fc_combined(obs)


        return self.fc_combined(obs.float())


class DQN_Training:

    def __init__(self):
        self.window = Window(8)
        self.env = Environment(8, 12, 5, self.window)


    def train(self):

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


        # Get number of actions from gym action space
        n_actions = 11
        # Get the number of state observations
        state = self.env.reset()
        n_observations = len(state)

        policy_net = DQN(n_actions)
        target_net = DQN(n_actions)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(50000)




        def select_action(obs, action_mask, steps_done):
         #   global steps_done
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
                 #   q_values[torch.tensor(action_mask) == 0] = -1e9
                #    action = q_values.argmax().item()
                    masked_q = q_values + torch.log(torch.from_numpy(action_mask).float() + 1e-8)
                    return torch.argmax(masked_q).item()
                 #   return action
            else:
                valid_actions = np.where(action_mask == 1)[0]
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
            non_final_mask = torch.tensor(
                [s is not None for s in batch.next_state],
                dtype=torch.bool
            )

            non_final_next_states = torch.stack([
                torch.FloatTensor(np.array([s]))  # Convert each list to tensor first
                for s in batch.next_state
                if s is not None  # Only include non-terminal states CHECK THIS
            ])

            state_batch = torch.cat([
                      torch.Tensor(np.array([s]))
                     for s in batch.state
            ])
           # state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32)

            action_batch = torch.cat([
                torch.LongTensor([a])
                for a in batch.action
            ])

            reward_batch = torch.cat([
                torch.FloatTensor([r])
                for r in batch.reward
            ])

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            q_values = policy_net(state_batch).squeeze(1)
            state_action_values = q_values.gather(1, action_batch.unsqueeze(1))

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(BATCH_SIZE)
            with torch.no_grad():
                next_q_values = target_net(non_final_next_states).squeeze(1)
               # print(next_q_values)
                max_next_q_values = next_q_values.max(1)[0]
                max_next_q_values = max_next_q_values.max(1)[0]
               # print(max_next_q_values)
                next_state_values[non_final_mask] = max_next_q_values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) # can also use nn.MSELoss

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

        steps_done = 0
        num_episodes = 5000
        rewards_list =[]

        for i_episode in range(num_episodes):

            obs, action_masks = self.env.reset()

            print("episode:", i_episode)
            done = False


            while not done:

                actions = {}

                for agent in self.env.agents:

                    agent_obs = torch.tensor(obs[agent], dtype=torch.float32)

                  #  print(agent_obs)
                 #   print(obs[agent])
                    action_mask = action_masks[agent]
                    actions[agent] = select_action(agent_obs, action_mask, steps_done)

                next_obs, rewards, terminated, truncated, action_masks = self.env.step(actions)

           #     print("rewards:", rewards)
                rewards_list.append(sum(rewards.values()))
                '''
                for agent in self.env.agents:

                    if terminated[agent]:
                        next_obs[agent]["observation"] = None
                    print(len(obs[agent]["observation"]))
                    memory.push(obs[agent]["observation"], actions[agent], next_obs[agent]["observation"], rewards[agent])

                '''

                for agent in self.env.agents:
                    memory.push(
                        [obs[agent]],
                        actions[agent],
                        None if terminated[agent] else [
                            next_obs[agent]
                        ],
                        rewards[agent]
                    )

                self.env.agents = [agent for agent in self.env.agents if not terminated[agent]]
               # for agent in self.env.agents:
               #     if terminated[agent]:
               #         self.env.agents.remove(agent)

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

        # Save the model
        torch.save(policy_net.state_dict(), 'DQN4.pth')
        with open('rewards4.pkl', 'wb') as f:
            pickle.dump(rewards_list, f)











