from typing import Any, Union, Optional
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
import environments.custom_lunar_lander_no_target

rng = np.random.default_rng(seed=42)

def _normalise(x):
        return x / np.sum(x)

def three_state_reward(state, target_state, weights):
    reward = -np.sqrt(np.sum(weights * (state[:3] - target_state[[0,1,4]]) ** 2))
    if reward > -0.01:
        reward += 10
    print("TW reward: ", reward)
    return reward

def three_state_obs(obs, target_state, weights):
    pos_obs = obs[[0,1,4]]  # only want x, y and angle
    pos_ts = target_state[[0,1,4]]
    # print((target_state - obs) * weights)
    # return (target_state - obs) * weights
    return np.concatenate((pos_ts - pos_obs, weights, obs[[2,3,5]]))

def target_state_fn(obs_space):
    # return rng.normal(loc=current_state[[0,1,4]], scale=0.2)
    low = np.array([-2.5, 0, -2*np.pi, 0, 0, 0], dtype=np.float32)
    high = np.array([2.5, 2.5, 2*np.pi, 1, 1, 1], dtype=np.float32)
    return rng.uniform(0.9*low, 0.9*high)

def weights_generation_fn():
    # return _normalise(rng.uniform(0, 1, size=(6,)))
    while True:
        # 50% chance to be non-zero
        mask = rng.random(3) < 0.5  # True with probability `chance`
        # generate numbers in [0,1] where mask is True, else 0
        result = mask * rng.random(3)
        if np.any(result != 0):
            return _normalise(result)

class Lunar_Lander_PPO_Env(gym.Wrapper):
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
    
    @staticmethod
    def make_env():
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = Lunar_Lander_PPO_Env(env, three_state_obs, three_state_reward,
                                target_state_fn, weights_generation_fn)
        return env
    
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