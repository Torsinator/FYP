import environments.custom_lunar_lander_no_target
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from typing import Any

class LunarLander9Obs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Update observation space
        low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[0,1,4]], env.observation_space.low[[2,3,5]]))
        high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[0,1,4]], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)

        self.target = np.array([0, 0, 0])
        print(self.target)
    
    @staticmethod
    def make_env():
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = LunarLander9Obs(env)
        return env
    
    def step(self, action):
        obs, reward, terminated, truncaded, info = self.env.step(action)
        obs = self.obs_fn(obs)
        return obs, reward, terminated, truncaded, info

    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return self.obs_fn(obs), info
    
    def obs_fn(self, obs) -> np.ndarray:
        """
        Convert observation(s) into a 1D or batch-friendly array.

        Works for:
        - Single obs: shape (obs_dim,)
        - Batched obs: shape (n_envs, obs_dim)
        Returns:
        - Single obs: shape (new_dim,)
        - Batched obs: shape (n_envs, new_dim)
        """
        obs = np.asarray(obs)

        # If batch dimension exists, process each row separately
        if obs.ndim == 2:
            # obs.shape = (n_envs, obs_dim)
            target = np.asarray(self.target).flatten()
            combined = np.concatenate([obs[:, [0, 1, 4]], 
                                    np.tile(target, (obs.shape[0], 1)), 
                                    obs[:, [2, 3, 5]]], axis=1)
        else:
            # Single observation
            target = np.asarray(self.target).flatten()
            combined = np.concatenate([obs[[0, 1, 4]], target, obs[[2, 3, 5]]])

        c = combined.astype(np.float32)
        return c


    
    def set_target_state(self, target):
        self.target = np.asarray(target).flatten()
    
    
class LunarLander6Obs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Update observation space
        low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[2,3,5]]))
        high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
    
    @staticmethod
    def make_env():
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = LunarLander9Obs(env)
        return env
    
    def step(self, action):
        return self.env.step(action)
    
class LunarLander9ObsWeighted(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Update observation space
        low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[0,1,4]], [0,0,0], env.observation_space.low[[2,3,5]]))
        high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[0,1,4]] [1,1,1], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
    
    @staticmethod
    def make_env():
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = LunarLander9Obs(env)
        return env
    
    def step(self, action):
        return self.env.step(action)
    
class LunarLander6ObsWeighted(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Update observation space
        low = np.concatenate((env.observation_space.low[[0,1,4]], [0,0,0], env.observation_space.low[[2,3,5]]))
        high = np.concatenate((env.observation_space.high[[0,1,4]], [1,1,1], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
    
    @staticmethod
    def make_env():
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = LunarLander9Obs(env)
        return env
    
    def step(self, action):
        return self.env.step(action)