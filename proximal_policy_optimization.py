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
from torch.optim import optimizer

from window import Window
from environment import Environment
import torch
import torch.nn as nn
import torch.nn.functional as F


class RolloutBuffer:
    def __init__(self, capacity, obs_size, action_size):
        self.capacity = capacity


        self.obs_buf = torch.zeros((self.capacity, obs_size))
        self.act_buf = torch.zeros((self.capacity, action_size))
        self.logprob_buf = torch.zeros((self.capacity,))
        self.rewards_buf = torch.zeros((self.capacity,))
        self.values_buf = torch.zeros((self.capacity,))
        self.truncated_buf = torch.zeros((self.capacity,))
        self.terminated_buf = torch.zeros((self.capacity,))

        self.idx = 0
        self.full = False

    def add(self, obs, action, logprob, reward, value, truncated, terminated):
        idx = self.idx
        self.obs_buf[self.idx] = obs
        self.act_buf[self.idx] = action
        self.logprob_buf[self.idx] = logprob
        self.rewards_buf[self.idx] = reward
        self.values_buf[self.idx] = value
        self.truncated_buf[self.idx] = truncated
        self.terminated_buf[self.idx] = terminated

        self.idx += 1
        if self.idx >= self.capacity:
            self.full = True
            self.idx = 0

    def get(self):

        length = self.capacity if self.full else self.idx

        data = dict(
            obs = self.obs_buf[:length],
            actions = self.act_buf[:length],
            logprobs = self.logprob_buf[:length],
            rewards = self.rewards_buf[:length],
            values = self.values_buf[:length],
            truncated = self.truncated_buf[:length],
            terminated = self.terminated_buf[:length]
        )

        self.idx = 0
        self.full = False
        return data


class PPO (nn.Module):

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

    def select_action_and_value(self, obs, action_mask):

        logits, value = self.forward(obs)
        logits = logits + torch.log(torch.from_numpy(action_mask).float() + 1e-8)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        #print(action)
        return action, dist.log_prob(action), dist.entropy(), value


class PPO_Training:

    def __init__(self):
        self.window = Window(8)
        self.env = Environment(8, 12, 5, self.window)


    def train(self):

        TIME = 2048
        GAMMA = 0.99
        GAE = 0.95
        CLIP = 0.2
        EPOCHS = 4
        BATCH_SIZE = 64
        LR = 3e-4

        obs_size = 658
        action_size = 11

        policy_net = PPO(action_size, obs_size)
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        buffer = RolloutBuffer(TIME, obs_size, action_size)


        def optimize_model(loader):

            for _ in range(EPOCHS):
                for batch_obs, batch_actions, batch_logprob, batch_returns, batch_advantages in loader:
                    logits, values = policy_net.forward(batch_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    batch_actions = batch_actions.argmax(dim=1)
                    new_logprob = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logprob - batch_logprob)
                    clip = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * batch_advantages
                    policy_loss = -torch.min(ratio * batch_advantages, clip).mean()

                    value_loss = F.mse_loss(values, batch_returns)

                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


        # steps_done = 0
        num_updates = 30
        obs, action_masks = self.env.reset()
        rewards_list =[]

        for i_update in range(1, num_updates+1):

            print("update: ", i_update)
           # obs, action_masks = self.env.reset()
            rewards_total = 0

            for t in range(TIME):


                actions = {}
                logprobs = {}
                entropies = {}
                values = {}

                for agent in self.env.agents:

                    agent_obs = torch.tensor(obs[agent], dtype=torch.float32)
                    action_mask = action_masks[agent]
                    #print("loc ",self.env.players[agent].get_x()," " ,self.env.players[agent].get_y())
                 #   print(len(self.env.agents))
                    actions[agent], logprobs[agent], entropies[agent], values[agent] = (
                            policy_net.select_action_and_value(agent_obs, action_mask))


                next_obs, rewards, terminated, truncated, action_masks = self.env.step(actions)
                print(sum(rewards.values()))
                rewards_total += sum(rewards.values())


                for agent in self.env.agents:

                    buffer.add(
                        torch.tensor(obs[agent], dtype=torch.float32),
                        actions[agent].item(),
                        logprobs[agent].detach(),
                        rewards[agent],
                        values[agent].detach(),
                        float(truncated[agent]),
                        float(terminated[agent])
                    )

                obs = next_obs

              #  for agent in self.env.agents:
             #       if terminated[agent]:
             #           self.env.agents.remove(agent)

                self.env.agents = [agent for agent in self.env.agents if not terminated[agent]]

                if all(list(truncated.values())) or len(self.env.agents) == 0:
                    print("end")
                    obs, action_masks = self.env.reset()

            rewards_list.append(rewards_total)

            data = buffer.get()

            obs = data["obs"]
            actions = data["actions"]
            logprobs = data["logprobs"]
            rewards = data["rewards"]
            values = data["values"]
            truncated = data["truncated"]
            terminated = data["terminated"]

            last_values = {}
            for agent in self.env.agents:
                agent_obs = torch.tensor(obs[agent], dtype=torch.float32)
                action_mask = action_masks[agent]
                with torch.no_grad():
                    _,_,_, last_values[agent] = policy_net.select_action_and_value(agent_obs, action_mask)

            last_values = torch.tensor([last_values[agent] for agent in self.env.agents], dtype=torch.float32)

            values = torch.cat([values, last_values], dim=0)

            advantages = torch.zeros_like(rewards)
            last_advantages = 0
            for t in reversed(range(TIME)):

                mask = 1.0 if not (terminated[t] or truncated[t]) else 0.0
                delta = rewards[t] + GAMMA * mask - values[t]
                advantages[t] = last_advantages = delta + GAMMA * GAE * mask * last_advantages

            returns = advantages + values[:-len(self.env.agents)]

            dataset = torch.utils.data.TensorDataset(obs, actions, logprobs, returns, advantages)
            loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            optimize_model(loader)


        # Save the model
        torch.save(policy_net.state_dict(), 'PPO2.pth')
        with open('rewards8.pkl', 'wb') as f:
            pickle.dump(rewards_list, f)

