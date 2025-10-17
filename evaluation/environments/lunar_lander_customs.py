import environments.custom_lunar_lander_no_target
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from typing import Any, Optional
from gymnasium import spaces

class LunarLander9Obs(gym.Wrapper):
    def __init__(self, env, weighted=False, dictionary_obs=False, dense_reward=False, rew_threshold=0.1, normalise_reward=False):
        super().__init__(env)

        self.weighted = weighted
        self.dictionary_obs = dictionary_obs
        self.dense_reward = dense_reward
        self.reward_threshold = rew_threshold

        # Update observation space
        if not weighted:
            low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[0,1,4]], env.observation_space.low[[2,3,5]]))
            high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[0,1,4]], env.observation_space.high[[2,3,5]]))
            self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
        else:
            low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[0,1,4]], [0,0,0], env.observation_space.low[[2,3,5]]))
            high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[0,1,4]], [1,1,1], env.observation_space.high[[2,3,5]]))
            self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
        
        if dictionary_obs:
            if not weighted:
                self.observation_space = spaces.Dict({
                "observation": self.observation_space,                  # full state (Box(8,))
                "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            })
            else:
                # TODO: Figure out what to do when weighted
                self.observation_space = spaces.Dict({
                "observation": self.observation_space,                  # full state (Box(8,))
                "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            })
        
        self.target = np.array([0, 0, 0])
        self.weights = np.array([0, 0, 0])
    
    @staticmethod
    def make_env(weighted=False, dictionary_obs=False, dense_reward=False):
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = LunarLander9Obs(env, weighted, dictionary_obs)
        return env
    
    def step(self, action):
        obs, reward, terminated, truncaded, info = self.env.step(action)
        obs = self.obs_fn(obs)
        if float(reward) < -1000:
            if self.dense_reward:
                reward = -1000
            else: 
                reward = -1
        else:
            reward = self.compute_reward(obs[[0,1,2]], self.target, info)
            if self.dense_reward and reward > self.reward_threshold:
                reward = 1000
                terminated = True
            elif self.dense_reward and reward == 1:
                terminated = True
        if self.dictionary_obs:
            obs = {
        "observation": obs.astype(np.float32),
        "achieved_goal": np.array(obs[[0,1,2]], dtype=np.float32),
        "desired_goal": np.array(self.target, dtype=np.float32)
        }
        return obs, reward, terminated, truncaded, info

    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if options is None:
            options = {}

        if "target_state" not in options:
            options["target_state"] = self.target_state_fn(self.env.observation_space)
        
        if "weights" not in options:
            options["weights"] = self.weights_fn()

        self.target_state = options["target_state"]
        self.weights = options["weights"]
        
        obs, info = super().reset(seed=seed, options=options)
        obs = self.obs_fn(obs)
        if self.dictionary_obs:
            obs = {
        "observation": obs.astype(np.float32),
        "achieved_goal": np.array(obs[[0,1,2]], dtype=np.float32),
        "desired_goal": np.array(self.target, dtype=np.float32)
        }
        return obs, info
    
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
        if isinstance(obs, dict):
            obs = obs["observation"]
        obs = np.asarray(obs)

        weights = np.asarray(self.weights).flatten()
        target = np.asarray(self.target).flatten()

        # If batch dimension exists, process each row separately
        if obs.ndim == 2:
            # obs.shape = (n_envs, obs_dim)
            if self.weighted:
                combined = np.concatenate([obs[:, [0, 1, 4]], 
                                        np.tile(target, (obs.shape[0], 1)), np.tile(weights, (obs.shape[0], 1)),
                                        obs[:, [2, 3, 5]]], axis=1)
            else:
                combined = np.concatenate([obs[:, [0, 1, 4]], 
                        np.tile(target, (obs.shape[0], 1)), 
                        obs[:, [2, 3, 5]]], axis=1)
        else:
            # Single observation
            if self.weighted:
                combined = np.concatenate([obs[[0, 1, 4]], target, obs[[2, 3, 5]]])
            else:
                combined = np.concatenate([obs[[0, 1, 4]], target, weights, obs[[2, 3, 5]]])

        c = combined.astype(np.float32)
        return c
    
    def set_target_state(self, target):
        self.target = np.asarray(target).flatten()
    
    def set_weights(self, weights):
        self.weights = np.asarray(weights).flatten()

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, _info: Optional[dict[str, Any]]
    ) -> np.ndarray:
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)

        # difference (works for (3,) or (batch, 3))
        diff = achieved_goal - desired_goal

        # optional weighting/scaling
        # if self.weights is shape (3,), this will broadcast fine
        if self.weighted:
            diff = self.weights * diff
        dist = np.linalg.norm(diff, axis=-1)

        # sparse reward example: 0 if within tolerance, -1 otherwise
        if self.dense_reward:
            return -dist
        
        return np.where(dist < self.reward_threshold, 1.0, 0).astype(np.float32) 
        # return dist

    def weights_fn(self):
        while True:
            mask = np.random.random(3) < 0.5
            result = mask * np.random.random(3)
            if np.any(result != 0):
                print(result / np.max(result))
                return result / np.max(result)
    
    def target_state_fn(self, obs_space):
        low = np.array([-1, -0.2, -2*np.pi])
        high = np.array([1, 1.5, 2*np.pi])
        return np.random.uniform(low, high)
    
class LunarLander6Obs(gym.Wrapper):
    def __init__(self, env, weighted=False, dictionary_obs=False, dense_reward=False, rew_threshold=0.1, normalise_reward=False):
        super().__init__(env)

        self.weighted = weighted
        self.dictionary_obs = dictionary_obs
        self.dense_reward = dense_reward
        self.reward_threshold = rew_threshold

        # Update observation space
        if not weighted:
            low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[2,3,5]]))
            high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[2,3,5]]))
            self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
        else:
            low = np.concatenate((env.observation_space.low[[0,1,4]], [0,0,0], env.observation_space.low[[2,3,5]]))
            high = np.concatenate((env.observation_space.high[[0,1,4]], [1,1,1], env.observation_space.high[[2,3,5]]))
            self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
        
        if dictionary_obs:
            if not weighted:
                self.observation_space = spaces.Dict({
                "observation": self.observation_space,                  # full state (Box(8,))
                "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            })
            else:
                # TODO: Figure out what to do when weighted
                self.observation_space = spaces.Dict({
                "observation": self.observation_space,                  # full state (Box(8,))
                "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            })
        
        self.target = np.array([0, 0, 0])
        self.weights = np.array([0, 0, 0])
    
    @staticmethod
    def make_env(weighted=False, dictionary_obs=False, dense_reward=False):
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = LunarLander6Obs(env, weighted, dictionary_obs)
        return env
    
    def step(self, action):
        obs, reward, terminated, truncaded, info = self.env.step(action)
        obs = self.obs_fn(obs)
        if float(reward) < -1000:
            if self.dense_reward:
                reward = -1000
            else: 
                reward = -1
        else:
            reward = self.compute_reward(obs[[0,1,2]], self.target, info)
            if self.dense_reward and reward > self.reward_threshold:
                reward = 1000
                terminated = True
            elif self.dense_reward and reward == 1:
                terminated = True
        if self.dictionary_obs:
            obs = {
        "observation": obs.astype(np.float32),
        "achieved_goal": np.array(obs[[0,1,2]], dtype=np.float32),
        "desired_goal": np.array(self.target, dtype=np.float32)
        }
        return obs, reward, terminated, truncaded, info

    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if options is None:
            options = {}

        if "target_state" not in options:
            options["target_state"] = self.target_state_fn(self.env.observation_space)
        
        if "weights" not in options:
            options["weights"] = self.weights_fn()

        self.target_state = options["target_state"]
        self.weights = options["weights"]
        
        obs, info = super().reset(seed=seed, options=options)
        obs = self.obs_fn(obs)
        if self.dictionary_obs:
            obs = {
        "observation": obs.astype(np.float32),
        "achieved_goal": np.array(obs[[0,1,2]], dtype=np.float32),
        "desired_goal": np.array(self.target, dtype=np.float32)
        }
        return obs, info
    
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
        if isinstance(obs, dict):
            obs = obs["observation"]
        obs = np.asarray(obs)

        weights = np.asarray(self.weights).flatten()
        target = np.asarray(self.target).flatten()

        # If batch dimension exists, process each row separately
        if obs.ndim == 2:
            # obs.shape = (n_envs, obs_dim)
            if self.weighted:
                combined = np.concatenate([obs[:, [0, 1, 4]] - 
                                        np.tile(target, (obs.shape[0], 1)), np.tile(weights, (obs.shape[0], 1)),
                                        obs[:, [2, 3, 5]]], axis=1)
            else:
                combined = np.concatenate([obs[:, [0, 1, 4]] - 
                        np.tile(target, (obs.shape[0], 1)), 
                        obs[:, [2, 3, 5]]], axis=1)
        else:
            # Single observation
            if self.weighted:
                combined = np.concatenate([obs[[0, 1, 4]] - target, obs[[2, 3, 5]]])
            else:
                combined = np.concatenate([obs[[0, 1, 4]] - target, weights, obs[[2, 3, 5]]])

        c = combined.astype(np.float32)
        return c
    
    def set_target_state(self, target):
        self.target = np.asarray(target).flatten()
    
    def set_weights(self, weights):
        self.weights = np.asarray(weights).flatten()

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, _info: Optional[dict[str, Any]]
    ) -> np.ndarray:
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)

        # difference (works for (3,) or (batch, 3))
        diff = achieved_goal[[0,1,2]]

        # optional weighting/scaling
        # if self.weights is shape (3,), this will broadcast fine
        if self.weighted:
            diff = self.weights * diff
        dist = np.linalg.norm(diff, axis=-1)

        # sparse reward example: 0 if within tolerance, -1 otherwise
        if self.dense_reward:
            return -dist
        
        return np.where(dist < self.reward_threshold, 1.0, 0).astype(np.float32) 
        # return dist

    def weights_fn(self):
        while True:
            mask = np.random.random(3) < 0.5
            result = mask * np.random.random(3)
            if np.any(result != 0):
                print(result / np.max(result))
                return result / np.max(result)
    
    def target_state_fn(self, obs_space):
        low = np.array([-1, -0.2, -2*np.pi])
        high = np.array([1, 1.5, 2*np.pi])
        return np.random.uniform(low, high)