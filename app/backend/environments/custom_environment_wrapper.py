from typing import Any
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class CustomEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, obs_fn, reward_fn, target_state_fn, weights_fn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.obs_fn = obs_fn
        self.target_state_fn = target_state_fn
        self.weights_fn = weights_fn
        self.weights = None
        self.target_state = None

        # Update observation space
        low = np.concatenate((env.observation_space.low[[0,1,4]], [0,0,0], env.observation_space.low[[2,3,5]]))
        high = np.concatenate((env.observation_space.high[[0,1,4]], [1,1,1], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.obs_fn(obs, self.target_state, self.weights)
        if float(reward) > -1000:
            reward = self.reward_fn(obs, self.target_state, self.weights)
        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if options is None:
            options = {}

        if "target_state" not in options:
            options["target_state"] = self.target_state_fn(self.env.observation_space)
        
        if "weights" not in options:
            options["weights"] = self.weights_fn()

        self.target_state = options["target_state"]
        self.weights = options["weights"]
        print("TW weights:", self.weights)

        obs, info = super().reset(seed=seed, options=options)
        obs = self.obs_fn(obs, self.target_state, self.weights)
        return obs, info
    
    def set_target_state(self, target_state):
        self.target_state = target_state
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_reward_function(self, reward_fn):
        self.reward_fn = reward_fn
    
    def set_obs_function(self, obs_fn):
        self.obs_fn = obs_fn
    