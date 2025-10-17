from typing import Any, Union, Optional
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

import environments.lunar_lander_customs as envs

num_envs = 24

def get_log_dir(classname):
    log_dir = f"training/{classname}/"
    os.makedirs(log_dir, exist_ok=True)

class PPO_Unweighted_9_States_Sparse():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=False)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=False)
        model = PPO.load(f"models/{cls.__name__}", env=env)
        return model, env

class PPO_Unweighted_9_States_Dense():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=False)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))        
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        return model, env
    
    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=True)
        model = PPO.load("models/PPO_Unweighted_6_States", env=env)
        return model, env
    
class PPO_Unweighted_6_States_Sparse():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=False)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))        
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = lambda: envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=False)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))        
        model = PPO.load("models/PPO_Unweighted_9_States", env=env)
        return model, env

class PPO_Unweighted_6_States_Dense():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=True)
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        return model, env
    
    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=True)
        model = PPO.load("models/PPO_Unweighted_6_States", env=env)
        return model, env

class PPO_Weighted_9_States_Sparse():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=False)
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=False)
        model = PPO.load("models/PPO_Unweighted_9_States", env=env)
        return model, env
    
class PPO_Weighted_9_States_Dense():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=True)
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=True)
        model = PPO.load("models/PPO_Unweighted_9_States", env=env)
        return model, env
    
class PPO_Weighted_6_States_Sparse():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=False)
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=False)
        model = PPO.load("models/PPO_Unweighted_9_States", env=env)
        return model, env
    
class PPO_Weighted_6_States_Dense():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=True)
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=True)
        model = PPO.load("models/PPO_Unweighted_9_States", env=env)
        return model, env
