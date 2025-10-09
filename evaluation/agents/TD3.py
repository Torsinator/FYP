from typing import Any, Union, Optional
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
from stable_baselines3 import TD3

import environments.lunar_lander_customs as envs

class PPO_Unweighted_9_States(TD3):
    def __init__(self) -> None:
        env = envs.LunarLander9Obs.make_env()
        self = TD3.load("./TD3_Unweighted_9_States", env=env)

class PPO_Unweighted_6_States(TD3):
    def __init__(self) -> None:
        env = envs.LunarLander6Obs.make_env()
        self = TD3.load("./TD3_Unweighted_6_States", env=env)

class PPO_Weighted_9_States(TD3):
    def __init__(self) -> None:
        env = envs.LunarLander9ObsWeighted.make_env()
        self = TD3.load("./TD3_Unweighted_6_States", env=env)