import functools
import random
import threading
import time
from math import floor
from math import ceil
from math import sqrt
import numpy as np
from gymnasium.spaces.space import Space
from pettingzoo.utils.env import AgentID
import player
from square import Square
import gym
from gymnasium import spaces
from player import Player
from pettingzoo import ParallelEnv
from copy import copy

from throw import Throw
from window import Window


class Environment(ParallelEnv):
   # metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size, players_count, balls, window):
        super().__init__()
        self.agent_selection = None
        self.court = None
        self.grid_size = grid_size
        self.players = [Player] * players_count
        self.balls = balls
        self.total_time = 100
        self.time = 0
        self.total_possession_time = 10
        self.possession_time_blue = 10
        self.possession_time_red = 10
        self.possible_agents = list(range(0, players_count))
        self.agents = copy(self.possible_agents)
        self.window = window
        self.render_mode = None
        court_size = 16 * 8 * 6  # grid_height * grid_width * channels
        team_size = len(self.players) * 1
        global_size = 2
        total_size = court_size + team_size + global_size

        self.observation_spaces = {
            agent_id: spaces.Box(low=0, high=1, shape=(total_size,), dtype=np.float32)
            for agent_id in self.agents
        }

        self.action_spaces = {
            agent_id: spaces.Discrete(5 + floor(players_count / 2))
            for agent_id in self.agents
        }




    def initialise_court(self):

        self.court = [[Square(0, None) for _ in range(self.grid_size)] for _ in range(self.grid_size * 2)]
        self.time = self.total_time
        self.possession_time_blue = self.total_possession_time
        self.possession_time_red = self.total_possession_time

        player_count = floor(len(self.players) / 2)
        step = (self.grid_size - 1) / (player_count - 1)

        for i in range(player_count):

            # Team Blue
            x_b = floor(i * step)
            y_b = 0
            new_player1 = Player(str(i),x_b, y_b, "blue")
            self.court[0][floor(i * step)].set_player(new_player1)

            # Team Red
            x_r = floor(i * step)
            y_r = (self.grid_size * 2) - 1
            new_player2 = Player(str(i + player_count), x_r, y_r, "red")
            self.court[len(self.court) - 1][floor(i * step)].set_player(new_player2)

        if self.balls % 2 == 0:
            blue_balls = floor(self.balls / 2)
            red_balls = blue_balls
        else:
            if random.random() < 0.5:
                blue_balls = floor(self.balls / 2)
                red_balls = floor(self.balls / 2) + 1
            else:
                blue_balls = floor(self.balls / 2) + 1
                red_balls = floor(self.balls / 2)

        self.initialise_balls(blue_balls, player_count, 0)
        self.initialise_balls(red_balls, player_count, len(self.court) - 1)
        self.get_players()
        #t1 = threading.Thread(target=self.solve)
        #self.window.initialise_players(self.court)


    def initialise_balls(self, balls, player_count, row):

        step = (player_count - 1) / (balls - 1)
        for i in range(balls):
            player_cur = 0
            for j in range(self.grid_size):
                if self.court[row][j].get_player() is not None:
                    if player_cur == ceil(i * step):
                        self.court[row][j].set_balls(1)
                        break
                    else:
                        player_cur += 1


    def get_players(self):

        count = 0

        for i in range(self.grid_size * 2):
            for j in range(self.grid_size):
                if self.court[i][j].get_player() is not None:
                    self.players[count] = self.court[i][j].get_player()
                    count += 1


    def get_observation(self, player):

        #self.get_players()
        channels = 6
        encoded_court = np.zeros((self.grid_size * 2, self.grid_size, channels), dtype=int)
        encoded_court[player.get_y()][player.get_x()][0] = 1

        for i in range(self.grid_size * 2):
            for j in range(self.grid_size):

                cur_square = self.court[i][j]
                encoded_court[i][j][4] = j / (self.grid_size - 1)
                encoded_court[i][j][5] = j / ((self.grid_size * 2) - 1)

                if cur_square.get_player() is not None:
                    if cur_square.get_player().get_alive():
                        if cur_square.get_player().get_team() == player.get_team():
                            encoded_court[i][j][1] = 1
                        else:
                            encoded_court[i][j][2] = 1

                if cur_square.get_balls() > 0:
                    encoded_court[i][j][3] = 1

        global_info = np.zeros(4)
        global_info[0] = self.time / self.total_time
        global_info[2] = self.get_team_balls(0) / 5.0
        global_info[3] = self.get_team_balls(1) / 5.0


        majority = self.ball_majority()
        if player.get_team() == "blue_team":
            global_info[1] = self.possession_time_blue / self.total_possession_time

        else:
            global_info[1] = self.possession_time_red / self.total_possession_time

        team_info = np.zeros((floor(len(self.players) / 2), 2))
        blue_team_alive = self.get_players_alive(0)
        red_team_alive = self.get_players_alive(1)

        blue_team = np.zeros(floor(len(self.players) / 2))
        for i in range(blue_team_alive):
            blue_team[i] = 1

        red_team = np.zeros(floor(len(self.players) / 2))
        for i in range(red_team_alive):
            red_team[i] = 1

        team_info_1 = blue_team if player.get_team() == "blue" else red_team
        team_info_2 = blue_team if player.get_team() == "red" else red_team


        team_info = np.concatenate((team_info_1, team_info_2), axis=0)
        all_info = np.concatenate((encoded_court.flatten(), team_info, global_info), axis=0)

        observation = ([encoded_court, team_info, global_info])
        return all_info

    def get_masked_observations(self):

        observations = {}

        for agent in self.agents:
            observation = self.get_observation(self.players[agent])
            action_mask = self.get_action_mask(self.players[agent])

            observation_mask = {
                agent: {
                    "observation": observation,
                    "action_mask": action_mask,
                }
            }
            observations.update(observation_mask)
        return observations


    def step(self, actions):

        rewards = [0] * len(actions)
        terminated = [False] * len(actions)

        players_name = list(actions.keys())

        i = 0
        throw_objects = []
        
        for action in actions.items():
            self.agent_selection = self.players[players_name[i]]
            if 5 <= action[1] < 5 + floor(len(self.players) / 2):
                throw_objects.append(self.get_throw_action(self.players[players_name[i]], action[1] - 5))
            else:
                rewards[i] = self.do_action(action[1], self.players[players_name[i]], players_name[i])
            i += 1

        throw_rewards = self.do_throws(throw_objects) # needs to return results to set agents to terminated and calc rewards

        remove = []

        for i in range(0, len(throw_rewards)):

            if throw_rewards[i] == -3:
                terminated[i] = True
                #self.window.remove_player(self.players[self.agents[i]].get_name())
                #self.agents.remove(self.agents[i])
                remove.append(self.agents[i])

            if throw_rewards[i] != 0:
                rewards[i] += throw_rewards[i]


        players_out = []
        if self.ball_majority() == 1:
            self.possession_time_blue -= 1
            if self.possession_time_blue <= 0:
                self.possession_time_blue = 10
                players_out = self.get_ball_holders(0)
                i = 0
                for agent in self.agents:
                    if agent > 6:
                        break
                    else:
                        rewards[i] -= 3

            #    print("out: ", players_out)
        else:
            self.possession_time_red -= 1
            if self.possession_time_red <= 0:
                self.possession_time_red = 10
                players_out = self.get_ball_holders(1)
                i = 0
                for agent in self.agents:
                    if agent >= 6:
                        rewards[i] -= 3

               # print("out: ", players_out)


        for p in players_out:
        #    print(rewards[self.agents.index(p)])
#            index = self.agents.index(p)
            self.players[p].set_alive(0)
            self.court[self.players[p].get_y()][self.players[p].get_x()].set_player(None)
            rewards[self.agents.index(p)] = -3
            terminated[self.agents.index(p)] = True
        #    print(rewards[self.agents.index(p)])

        remove.extend(players_out)
        players_left = list(set(self.agents) - set(remove))
        players_left.sort()
     #   print(players_left)
        rewards_dict = dict(zip(players_name, rewards))

        #observations = self.get_masked_observations()
        infos = dict((name, {}) for name in self.agents)

        if len(players_left) > 0:
            #print("left ",players_left)

            if players_left[0] >= floor(len(self.players) / 2) or players_left[len(players_left) - 1] < floor(len(self.players) / 2):
                truncations = dict((name,True) for name in players_name)
                print(self.agents)
                print(remove)
                print(players_left)
                for p in players_left:
                    rewards_dict[p] = 5

            else:
                truncations = dict((name, False) for name in players_name)

        else:
            truncations = dict((name, False) for name in players_name)

        self.time -= 1

        if self.time == 0:
            truncations = dict((name, True) for name in players_name)
            blue = 0
            red = 0
            for agent in self.agents:
                if self.players[agent].get_team() == "blue":
                    blue += 1
                else:
                    red += 1

            for agent in self.agents:
                if blue == red:
                    rewards_dict[agent] -= 5
                    print("draw")
                elif self.players[agent].get_team() == "blue":
                    print("blue ",players_left)
                    rewards_dict[agent] -= 5 if blue > red else red > blue -5
                else:
                    print("red ",players_left)
                    rewards_dict[agent] -= 5 if red > blue else blue > red -5


        observations = {}
        action_masks = {}
        for agent in self.agents:
            observation = {agent: self.get_observation(self.players[agent])}
            observations.update(observation)
            action_mask = {agent: self.get_action_mask(self.players[agent])}
            action_masks.update(action_mask)


        #observations_dict = dict(zip(players_name, observations))
        terminated_dict = dict(zip(players_name, terminated))
        #self.window.update_balls()

        #self.window.render(self)

        return [observations, rewards_dict, terminated_dict, truncations, action_masks]


    def do_action(self, action, player, agent):

        reward = 0.0

        if action == 0:
            return -0.5
        if action == 1:
            reward = self.move_player(player, player.get_x(), player.get_y(), player.get_x(), player.get_y()+1, agent)
            if self.court[player.get_y()][player.get_x()].get_balls() > 0:
                reward += 0.1
            return 0.1 + reward
        elif action == 2:
            reward = self.move_player(player, player.get_x(), player.get_y(), player.get_x(), player.get_y() - 1, agent)
            if self.court[player.get_y()][player.get_x()].get_balls() == 0:
                reward += 0.1
            return 0.1 + reward
        elif action == 3:
            reward = self.move_player(player, player.get_x(), player.get_y(), player.get_x() + 1, player.get_y(), agent)
            return 0.05 + reward
        elif action == 4:
            reward = self.move_player(player, player.get_x(), player.get_y(), player.get_x() - 1, player.get_y(), agent)
            return 0.05 + reward
     #   elif action >= 5 & action < 5 + floor(len(self.players) / 2):
     #       self.throw_ball(player, action - 5)

        return 1

    def move_player(self, player, old_x, old_y, new_x, new_y, agent):

        reward = 0.0
        new_square = self.court[new_y][new_x]
        if new_square.get_player() is None:
            old_square = self.court[old_y][old_x]
            player.set_x(new_x)
            player.set_y(new_y)
            old_square.set_player(None)
            new_square.set_player(player)
            if new_square.get_balls() > 0:  # increases reward for picking up ball
                reward = 0.8
            new_square.set_balls(old_square.get_balls() + new_square.get_balls())
            old_square.set_balls(0)
        return reward

           # if old_x != new_x:
           #     self.window.update_player(agent, (new_x - old_x), 0)
          #  else:
          #      self.window.update_player(agent, 0, (new_y - old_y))

         #   return reward


    def get_action_mask(self, player):

        action_mask = [1] * (5 + floor(len(self.players) / 2))
        action_mask = np.ones(11, dtype=np.int8)

        if player.get_y() >= (self.grid_size * 2) - 1:
            action_mask[1] = 0

        elif ((self.court[player.get_y() + 1][player.get_x()].get_player() is not None) |
              (floor(player.get_y() / self.grid_size)) != (floor((player.get_y() + 1) / self.grid_size))) :

            action_mask[1] = 0

        if player.get_y() == 0:
            action_mask[2] = 0

        elif ((self.court[player.get_y() - 1][player.get_x()].get_player() is not None) |
              (floor(player.get_y() / self.grid_size)) != (floor((player.get_y() - 1) / self.grid_size))):
            action_mask[2] = 0

        if player.get_x() >= self.grid_size - 1:
            action_mask[3] = 0
        elif self.court[player.get_y()][player.get_x() + 1].get_player() is not None:
            action_mask[3] = 0

        if player.get_x() == 0:
            action_mask[4] = 0
        elif self.court[player.get_y()][player.get_x() - 1].get_player() is not None:
            action_mask[4] = 0

        targets = 0
        if self.court[player.get_y()][player.get_x()].get_balls()  > 0:
            if player.get_team() == "blue":
                targets = self.get_players_alive(1)
            else:
                targets = self.get_players_alive(0)

        for i in range(0, (floor(len(self.players) / 2) - targets)):
            action_mask[4 + targets + i + 1] = 0

        return action_mask


    def get_throw_action(self, player, target):

        count = 0

        for i in range(0, self.grid_size):
            if player.get_team() == "blue":
                for j in range(self.grid_size, self.grid_size * 2):
                    if self.court[j][i].get_player() is not None:
                        if count == target:
                            return Throw(player, self.court[j][i].get_player())
                        count += 1
            else:
                for j in range(self.grid_size, -1, -1):
                    if self.court[j][i].get_player() is not None:
                        if count == target:
                            return Throw(player, self.court[j][i].get_player())
                        count += 1

        return None


    def do_throws(self, throws):

        throws_grouped = []

        for i in range(0, len(throws)):
            if throws[i] is None:
                continue
            same = False
            for j in range(0, len(throws_grouped)):
                if throws_grouped[j][0] is None:
                    continue
                if throws_grouped[j][0].get_target() == throws[i].get_target():
                    throws_grouped[j].append(throws[i])
                    same = True
                    break

            if not same:
                throws_grouped.append([throws[i]])

        rewards = [0.0] * len(self.agents)

        for i in range(0, len(throws_grouped)):

            if throws_grouped[i][0] is None:
                continue
            if throws_grouped[i][0].get_thrower().get_team() == "blue":
                self.possession_time_blue = 10
            else:
                self.possession_time_red = 10

            chance = self.calc_hit_chance(throws_grouped[i])

            rand = random.uniform(0,1)
            target = throws_grouped[i][0].get_target()
            square = self.court[target.get_y()][target.get_x()]
            hit = False
            if rand <= chance:
                hit = True
                rewards[self.agents.index(int(target.get_name()))] = -3
                target.set_alive(0)
                square.set_player(None)

            square.set_balls(square.get_balls() + len(throws_grouped[i]))

            for j in range(0, len(throws_grouped[i])):
                thrower = throws_grouped[i][j].get_thrower()
                square = self.court[thrower.get_y()][thrower.get_x()]
                square.set_balls(square.get_balls() - 1)

                if rewards[self.agents.index(int(thrower.get_name()))] != -3:
                    if hit:
                        rewards[self.agents.index(int(thrower.get_name()))] = 10
                    else:
                        rewards[self.agents.index(int(thrower.get_name()))] = -0.2

        return rewards



    def calc_hit_chance(self, throws):

        max_dist = (self.grid_size * 2) - 1
        balls = len(throws)
        dist = self.grid_size * 3

        for i in range(0, len(throws)):
            cur_dist = self.dist_two_players(throws[i].get_target(), throws[i].get_thrower())
            if cur_dist < dist:
                dist = cur_dist


        if dist <= floor(max_dist * 0.3):
            return 1

        if dist <= floor(max_dist * 0.4):
            if balls > 1:
                return 0.9
            return 0.8

        if dist <= ceil(max_dist * 0.5):
            if balls > 1:
                return 0.6 * (1.1 ** balls)
            return 0.6

        if dist <= floor(max_dist * 0.7):
            if balls > 1:
                return 0.4 * (1.1 ** balls)
            return 0.4

        if dist <= floor(max_dist * 0.8):
            if balls > 1:
                return 0.3 * (1.1 ** balls)
            return 0.3

        if dist <= floor(max_dist * 0.9):
            if balls > 1:
                return 0.1 * (1.1 ** balls)
            return 0.1

        return 0


    def dist_two_players(self, player1, player2):

        return floor(sqrt( abs(player1.get_x() - player2.get_x()) ** 2 + abs(player1.get_y() - player2.get_y()) ** 2 ))


    def reset(self, seed=None, options=None):
        self.initialise_court()
        self.agents = copy(self.possible_agents)

        #observations = self.get_masked_observations()
        observations = {}
        action_masks = {}
        for agent in self.agents:
            observation = {agent: self.get_observation(self.players[agent])}
            observations.update(observation)
            action_mask = {agent: self.get_action_mask(self.players[agent])}
            action_masks.update(action_mask)

        infos = dict((name, {}) for name in self.agents)

        return observations, action_masks

    def get_players_alive(self, team):
        count = 0

        for i in range(floor(len(self.players)/2)):
            if self.players[i + team * floor(len(self.players) / 2)].get_alive():
                count += 1
        return count


    def get_ball_holders(self, team):

        players = []

        for i in range(team, self.grid_size * (team + 1)):
            for j in range(self.grid_size):
                cur_square = self.court[i][j]
                if cur_square.get_player() is not None:
                    if cur_square.get_balls() > 0:
                        players.append(int(cur_square.get_player().get_name()))

        return players


    def ball_majority(self):

        team_blue_balls = 0
        team_red_balls = 0

        for i in range(self.grid_size * 2):
            for j in range(self.grid_size):
                cur_square = self.court[i][j]
                if cur_square.get_balls() > 0:
                    if i >= self.grid_size:
                        team_red_balls += cur_square.get_balls()
                    else:
                        team_blue_balls += cur_square.get_balls()

        if team_blue_balls > team_red_balls:
            return 1
        return 2


    def get_team_balls(self, team):
        balls = 0

        for i in range((self.grid_size * team), self.grid_size * (1 + team)):
            for j in range(self.grid_size):
                cur_square = self.court[i][j]
                if cur_square.get_balls() > 0:

                    balls += cur_square.get_balls()

        return balls



    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):

        return self.observation_spaces[agent]



    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]


