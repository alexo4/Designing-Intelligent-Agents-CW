#from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
import threading

from gym import spaces
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module import RLModule, RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune import register_env
import ray
from environment import Environment
from reinforcement_learning import DQN_Training
from window import Window
import tkinter as tk
from pettingzoo.test import parallel_api_test
#from stable_baselines3 import PPO
#from stable_baselines3.ppo import MlpPolicy
#import supersuit as ss
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import numpy as np
from math import floor
import torch
import torch.nn as nn
#from ray.rllib.connectors.env_to_module.default_env_to_module import DefaultEnvToModuleConnector

def main():



    #window = Window(8)
    env = Environment(8, 12, 5, window)
   # env.reset()
    #window.start_mainloop()
    dqn = DQN_Training()

    #t1 = threading.Thread(target=dqn.train(), args=(env,))
    #t1.start()

    #window.start_mainloop()
    #parallel_api_test(env)



    #environment = Environment(8, 12, 5)
    #environment.initialise_court()

    #window = Window(environment.court)
    print("hi")

main()
'''
def model(window):

    env = Environment(8, 12, 5, window)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    #env = ss.concat_vec_env_v1(env, 1, base_class="stable_baselines3")

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=0.01,
        batch_size = 256,
    )

    model.learn(total_timesteps=10000)

    model.save(f"DAI-model{time.strftime('%Y%m%d-%H%M%S')}")
    env.close()
'''
print("hi")
#window = Window(8)





