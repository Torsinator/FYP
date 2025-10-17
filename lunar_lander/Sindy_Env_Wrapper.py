from typing import Any, Union, Optional
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
import custom_lunar_lander_no_target

class LunarLanderSindy(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Update observation space
        low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[2,3,5]]))
        high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
    
    @staticmethod
    def make_env():
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = LunarLanderSindy(env)
        return env
    
    def step(self, action):
        obs, reward, terminated, truncaded, info = self.env.step(action)
        obs = self.gym_state_to_mpc(obs)
        return obs, reward, terminated, truncaded, info

    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return self.gym_state_to_mpc(obs), info
        
    # Map Gym state to MPC state format
    def gym_state_to_mpc(self, state):
        x, y, vx, vy, theta, omega, *_ = state
        # Multipliers from Lunar Lander
        return np.array([x * 10, y * 6.666, vx * 5, vy * 7.5, theta, omega * 2.5])

    