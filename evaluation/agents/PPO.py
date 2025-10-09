from typing import Any, Union, Optional
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

import environments.lunar_lander_customs as envs

class PPO_Unweighted_9_States():
    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env()
        model = PPO.load("models/PPO_Unweighted_9_States", env=env)
        return model, env

class PPO_Unweighted_6_States(PPO):
    def __init__(self) -> None:
        env = envs.LunarLander6Obs.make_env()
        self = PPO.load("models/PPO_Unweighted_6_States", env=env)

class PPO_Weighted_9_States(PPO):
    def __init__(self) -> None:
        env = envs.LunarLander9ObsWeighted.make_env()
        self = PPO.load("models/PPO_Unweighted_6_States", env=env)