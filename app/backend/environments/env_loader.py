import gymnasium as gym
import environments.custom_lunar_lander_no_target
from environments.lunar_lander_sac_her_env import Lunar_Lander_SAC_HER_Env
from environments.lunar_lander_ppo_env import Lunar_Lander_PPO_Env
import numpy as np

envs = [Lunar_Lander_SAC_HER_Env, Lunar_Lander_PPO_Env]

def get_env(env_name : str):
    for env in envs:
        if env.__name__ == env_name:
            return env.make_env()
    raise LookupError(f"Could not find env {env_name}")
