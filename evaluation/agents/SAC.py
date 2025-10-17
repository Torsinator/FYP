from typing import Any, Union, Optional
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC

import environments.lunar_lander_customs as envs

sac_model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=16,
            goal_selection_strategy=goal_selection_strategy,
        ),
        learning_starts=(num_envs)*50*20*10,
        verbose=1,
    )

class SAC_Unweighted_9_States_Sparse():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=False)
        model = SAC("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=False)
        model = SAC.load("models/SAC_Unweighted_9_States", env=env)
        return model, env

class SAC_Unweighted_9_States_Dense():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=True)
        model = SAC("MlpPolicy", env, verbose=1, device="cpu")
        return model, env
    
    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=True)
        model = SAC.load("models/SAC_Unweighted_6_States", env=env)
        return model, env
    
class SAC_Unweighted_6_States_Sparse():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=False)
        model = SAC("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=False)
        model = SAC.load("models/SAC_Unweighted_9_States", env=env)
        return model, env

class SAC_Unweighted_6_States_Dense():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=True)
        model = SAC("MlpPolicy", env, verbose=1, device="cpu")
        return model, env
    
    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=False, dense_reward=True)
        model = SAC.load("models/SAC_Unweighted_6_States", env=env)
        return model, env

class SAC_Weighted_9_States_Sparse():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=False)
        model = SAC("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=False)
        model = SAC.load("models/SAC_Unweighted_9_States", env=env)
        return model, env
    
class SAC_Weighted_9_States_Dense():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=True)
        model = SAC("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=True)
        model = SAC.load("models/SAC_Unweighted_9_States", env=env)
        return model, env
    
class SAC_Weighted_6_States_Sparse():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=False)
        model = SAC("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=False)
        model = SAC.load("models/SAC_Unweighted_9_States", env=env)
        return model, env
    
class SAC_Weighted_6_States_Dense():
    @classmethod
    def create_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=True)
        model = SAC("MlpPolicy", env, verbose=1, device="cpu")
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=False, dense_reward=True)
        model = SAC.load("models/SAC_Unweighted_9_States", env=env)
        return model, env
