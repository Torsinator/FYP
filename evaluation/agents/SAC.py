import os
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_util import make_vec_env

import environments.lunar_lander_customs as envs

num_envs = 24
def get_log_dir(classname):
    log_dir = f"training/{classname}/"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

import numpy as np

def lr_schedule(initial_lr=3e-4, warmup_frac=0.05, final_frac=0.2):
    """
    Cosine LR schedule with warmup and smooth decay.
    - warmup_frac: % of total steps used to ramp up
    - final_frac:  minimum fraction of initial LR at end
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining = 1 → start, 0 → end
        progress = 1.0 - progress_remaining
        warmup_steps = warmup_frac
        if progress < warmup_steps:
            # linear warmup from 0 → initial_lr
            lr = initial_lr * (progress / warmup_steps)
        else:
            # cosine decay from initial_lr → final_lr
            decay_progress = (progress - warmup_steps) / (1 - warmup_steps)
            cosine = 0.5 * (1 + np.cos(np.pi * decay_progress))
            lr = initial_lr * (final_frac + (1 - final_frac) * cosine)
        return lr
    return func

def create_SAC_model(env):
    return SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=16,
                goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            ),
            learning_starts=(num_envs + 1)*50*20,   
            verbose=1,
        )

class SAC_Unweighted_9_States_Sparse():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=True, dense_reward=False)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))
        model = create_SAC_model(env)
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=True, dense_reward=False)
        model = SAC.load(f"models/{cls.__name__}", env=env)
        return model, env

class SAC_Unweighted_9_States_Dense():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=True, dense_reward=True)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))
        model = create_SAC_model(env)
        return model, env
    
    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=False, dictionary_obs=True, dense_reward=True)
        model = SAC.load(f"models/{cls.__name__}", env=env)
        return model, env
    
class SAC_Unweighted_6_States_Sparse():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=True, dense_reward=False)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))
        model = create_SAC_model(env)
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=True, dense_reward=False)
        model = SAC.load(f"models/{cls.__name__}", env=env)
        return model, env

class SAC_Unweighted_6_States_Dense():
    @classmethod
    def create_model(cls):
        env = lambda:envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=True, dense_reward=True)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))
        model = create_SAC_model(env)
        return model, env
    
    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=False, dictionary_obs=True, dense_reward=True)
        model = SAC.load(f"models/{cls.__name__}", env=env)
        return model, env

class SAC_Weighted_9_States_Sparse():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=True, dense_reward=False)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))
        model = create_SAC_model(env)
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=True, dense_reward=False)
        model = SAC.load(f"models/{cls.__name__}", env=env)
        return model, env
    
class SAC_Weighted_9_States_Dense():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=True, dense_reward=True)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))
        model = create_SAC_model(env)
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander9Obs.make_env(weighted=True, dictionary_obs=True, dense_reward=True)
        model = SAC.load(f"models/{cls.__name__}", env=env)
        return model, env
    
class SAC_Weighted_6_States_Sparse():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=True, dense_reward=False)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))
        model = create_SAC_model(env)
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=True, dense_reward=False)
        model = SAC.load(f"models/{cls.__name__}", env=env)
        return model, env
    
class SAC_Weighted_6_States_Dense():
    @classmethod
    def create_model(cls):
        env = lambda: envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=True, dense_reward=True)
        env = make_vec_env(env, num_envs, monitor_dir=get_log_dir(cls.__name__))
        model = create_SAC_model(env)
        return model, env

    @classmethod
    def load_model(cls):
        env = envs.LunarLander6Obs.make_env(weighted=True, dictionary_obs=True, dense_reward=True)
        model = SAC.load(f"models/{cls.__name__}", env=env)
        return model, env
