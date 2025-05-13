from math import floor

import numpy as np
import torch

from deep_q_learning import DQN
from environment import Environment
from proximal_policy_optimization import PPO
from window import Window


class Testing:

    def __init__(self):
        self.model_DQN = "DQN"
        self.model_PPO = "PPO"
        self.model_RAND = "RAND"
        self.window = Window(8)
        self.grid_size = 8
        self.players_count = 12
        self.balls_count = 5


    def game(self, model_one, model_two, rounds):

        counts = [0] * 3

        if model_one != self.model_RAND:
            model_one = self.load_model(model_one)

        if model_two != self.model_RAND:
            model_two = self.load_model(model_two)

        for i in range(rounds):
            counts[self.run_round(model_one, model_two) - 1] += 1

        print(counts)


    def run_round(self, model_one, model_two):

        env = Environment(self.grid_size, self.players_count, self.balls_count, self.window)

        obs, action_masks = env.reset()

        done = False

        while not done:

            actions = {}

            for agent in env.agents:
                agent_obs = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
                action_mask = action_masks[agent]

                if agent < floor(self.players_count / 2):
                    actions[agent] = self.select_action(agent_obs, action_mask, model_one)
                else:
                    actions[agent] = self.select_action(agent_obs, action_mask, model_two)

            next_obs, rewards, terminated, truncated, action_masks = env.step(actions)

            for agent in env.agents:
                if terminated[agent]:
                    env.agents.remove(agent)

            obs = next_obs

            if all(list(truncated.values())):

                done = True

        # return 1 for model 1 win, 2 for model 2 win and 3 for draw

        team_one = 0
        team_two = 0

        for agent in env.agents:
            if agent < floor(self.players_count / 2):
                team_one += 1
            else:
                team_two += 1

        if team_one > team_two:
            return 1
        elif team_one < team_two:
            return 2
        else:
            return 3



    def select_action(self, obs, action_mask, model):

        if model == self.model_RAND:
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions)
        elif model == self.model_DQN:

            with torch.no_grad():
                q_values = model.forward(obs)
                masked_q = q_values + torch.log(torch.from_numpy(action_mask).float() + 1e-8)
                return torch.argmax(masked_q).item()

        elif model == self.model_PPO:
            with torch.no_grad():
                action,_,_,_ = model.select_action_and_value(obs, action_mask)
                return action
        valid_actions = np.where(action_mask == 1)[0]
        return np.random.choice(valid_actions)


    def load_model(self, model):

        if model == self.model_DQN:
            model = DQN(11)
            model.load_state_dict(torch.load("DQN6.pth"))
            model.eval()
        elif model == self.model_PPO:
            model = PPO(11, 784)
            model.load_state_dict(torch.load("PPO2.pth"))
            model.eval()

       # model = DQN(11)
      #  model.load_state_dict(torch.load("DQN.pth"))
      #  model.eval()

        return model










