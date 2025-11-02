import environments.custom_lunar_lander_no_target
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from typing import Any, Optional
from gymnasium import spaces

class LunarLander9Obs(gym.Wrapper):
    def __init__(self, env, weighted=False, dictionary_obs=False, dense_reward=False, rew_threshold=0.3, normalise_reward=False):
        super().__init__(env)

        self.weighted = weighted
        self.dictionary_obs = dictionary_obs
        self.dense_reward = dense_reward
        self.reward_threshold = rew_threshold

        # Update observation space
        if not weighted:
            if dictionary_obs:
                low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[2,3,5]]))
                high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[2,3,5]]))
            else:
                low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[0,1,4]], env.observation_space.low[[2,3,5]]))
                high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[0,1,4]], env.observation_space.high[[2,3,5]]))
        else:
            if dictionary_obs:
                low = np.concatenate((env.observation_space.low[[0,1,4]], [0,0,0], env.observation_space.low[[2,3,5]]))
                high = np.concatenate((env.observation_space.high[[0,1,4]], [1,1,1], env.observation_space.high[[2,3,5]]))
            else:
                low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[0,1,4]], [0,0,0], env.observation_space.low[[2,3,5]]))
                high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[0,1,4]], [1,1,1], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
        
        if dictionary_obs:
            if not weighted:
                self.observation_space = spaces.Dict({
                "observation": self.observation_space,                  # full state (Box(8,))
                "desired_goal": Box(low=low[:3], high=high[:3], dtype=env.observation_space.dtype),
                "achieved_goal": Box(low=low[:3], high=high[:3], dtype=env.observation_space.dtype),
            })
            else:
                # TODO: Figure out what to do when weighted
                self.observation_space = spaces.Dict({
                "observation": self.observation_space,                  # full state (Box(8,))
                "desired_goal": Box(low=low[:3], high=high[:3], dtype=env.observation_space.dtype),
                "achieved_goal": Box(low=low[:3], high=high[:3], dtype=env.observation_space.dtype),
            })
        
        self.target = np.array([0, 0, 0])
        self.weights = np.array([0, 0, 0])
    
    @staticmethod
    def make_env(weighted=False, dictionary_obs=False, dense_reward=False):
        env = gym.make("CustomLunarLander-v0", render_mode=None, continuous=True)
        env = LunarLander9Obs(env, weighted, dictionary_obs, dense_reward)
        return env
    
    def step(self, action):
        obs, reward, terminated, truncaded, info = self.env.step(action)
        obs = self.obs_fn(obs)
        if self.weighted and self.dictionary_obs:
            info["weights"] = obs[3:6]
        elif self.weighted:
            info["weights"] = obs[6:9]
        achieved_goal = obs[:3]
        desired_goal = self.target
        
        if float(reward) < -1000:
            if self.dense_reward:
                reward = -1000
            else: 
                reward = -1
        else:
            reward = self.compute_reward(achieved_goal, desired_goal, info)
            if self.dense_reward and reward > -self.reward_threshold:
                reward = 1000
                terminated = True
            elif not self.dense_reward and reward == 1:
                terminated = True
            # elif not self.dictionary_obs:
            #     if self.last_dist is None:
            #         self.last_dist = -reward
            #         reward = 0
            #     else:
            #         reward, self.last_dist = self.last_dist + reward, -reward
        if self.dictionary_obs:
            obs = {
        "observation": obs.astype(np.float32),
        "achieved_goal": achieved_goal.astype(np.float32),
        "desired_goal": desired_goal.astype(np.float32)
        }
        return obs, reward, terminated, truncaded, info

    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if options is None:
            options = {}

        if "target_state" not in options:
            options["target_state"] = self.target_state_fn(self.env.observation_space)
        
        if "weights" not in options:
            options["weights"] = self.weights_fn()

        self.target = options["target_state"]
        self.weights = options["weights"]
        self.last_dist = None
        
        obs, info = super().reset(seed=seed, options=options)
        obs = self.obs_fn(obs)
        if self.weighted and self.dictionary_obs:
            info["weights"] = obs[3:6]
        elif self.weighted:
            info["weights"] = obs[6:9]
        achieved_goal = obs[:3]
        desired_goal = self.target
        assert self.target.shape == (3,), f"target shape = {self.target.shape}"
        assert self.weights.shape == (3,), f"weights shape = {self.weights.shape}"
        # print(obs)
        if self.dictionary_obs:
            obs = {
        "observation": obs.astype(np.float32),
        "achieved_goal": achieved_goal.astype(np.float32),
        "desired_goal": desired_goal.astype(np.float32)
        }
        # print("NEW episode")
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

        if self.dictionary_obs:
            target = weights = np.array([])
            if self.weighted:
                weights = self.weights
        else:
            target = np.asarray(self.target).flatten()
            weights = np.asarray(self.weights).flatten()
            assert target.shape == (3,), f"Bad target shape {target.shape}"
            assert weights.shape == (3,), f"Bad weights shape {weights.shape}"

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
                combined = np.concatenate([obs[[0, 1, 4]], target, weights, obs[[2, 3, 5]]])
            else:
                combined = np.concatenate([obs[[0, 1, 4]], target, obs[[2, 3, 5]]])

        c = combined.astype(np.float32)
        return c
    
    def set_target_state(self, target):
        self.target = np.asarray(target).flatten()
    
    def set_weights(self, weights):
        self.weights = np.asarray(weights).flatten()

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Optional[dict[str, Any]]
    ) -> np.ndarray:
        # assert np.allclose(achieved_goal[3:6], desired_goal[3:6]), "weights mismatch"
        diff = achieved_goal - desired_goal

        if not self.weighted:
            dist = np.linalg.norm(diff, axis=-1)
        else:
            assert(info is not None)
            if isinstance(info, list):
                info = info[0]
            dist = np.sum((info["weights"] * diff**2))

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
                # print(result / np.max(result))
                return result / np.max(result)
    
    def target_state_fn(self, obs_space):
        # print("TW tragets")
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
        else:
            low = np.concatenate((env.observation_space.low[[0,1,4]], [0,0,0], env.observation_space.low[[2,3,5]]))
            high = np.concatenate((env.observation_space.high[[0,1,4]], [1,1,1], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
        
        if dictionary_obs:
            if not weighted:
                self.observation_space = spaces.Dict({
                "observation": self.observation_space,                  # full state (Box(8,))
                "desired_goal": Box(low=low[:3], high=high[:3], dtype=env.observation_space.dtype),
                "achieved_goal": Box(low=low[:3], high=high[:3], dtype=env.observation_space.dtype),
            })
            else:
                # TODO: Figure out what to do when weighted
                self.observation_space = spaces.Dict({
                "observation": self.observation_space,                  # full state (Box(8,))
                "desired_goal": Box(low=low[:3], high=high[:3], dtype=env.observation_space.dtype),
                "achieved_goal": Box(low=low[:3], high=high[:3], dtype=env.observation_space.dtype),
            })
        
        self.target = np.array([0, 0, 0])
        self.weights = np.array([0, 0, 0])
    
    @staticmethod
    def make_env(weighted=False, dictionary_obs=False, dense_reward=False):
        env = gym.make("CustomLunarLander-v0", render_mode=None, continuous=True)
        env = LunarLander6Obs(env, weighted, dictionary_obs, dense_reward)
        return env
    
    def step(self, action):
        obs, reward, terminated, truncaded, info = self.env.step(action)
        obs = self.obs_fn(obs)
        if self.weighted:
            info["weights"] = obs[3:6]
        achieved_goal = obs[:3]
        desired_goal = self.target
        if self.weighted:
            achieved_goal = np.concatenate([achieved_goal, self.weights])
            desired_goal = np.concatenate([desired_goal, self.weights])
        
        if float(reward) < -1000 or truncaded:
            if self.dense_reward:
                reward = -1000
            else: 
                reward = -1
        else:
            reward = self.compute_reward(achieved_goal, desired_goal, info)
            if self.dense_reward and reward > -self.reward_threshold:
                reward = 1000
                terminated = True
            elif not self.dense_reward and reward == 1:
                terminated = True
        if self.dictionary_obs:
            obs = {
        "observation": obs.astype(np.float32),
        "achieved_goal": achieved_goal.astype(np.float32),
        "desired_goal": desired_goal.astype(np.float32)
        }
        return obs, reward, terminated, truncaded, info

    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if options is None:
            options = {}

        if "target_state" not in options:
            options["target_state"] = self.target_state_fn(self.env.observation_space)
        
        if "weights" not in options:
            options["weights"] = self.weights_fn()

        self.target = options["target_state"]
        self.weights = options["weights"]
        
        obs, info = super().reset(seed=seed, options=options)
        obs = self.obs_fn(obs)
        achieved_goal = obs[:3]
        desired_goal = self.target
        if self.weighted:
            achieved_goal = np.concatenate([achieved_goal, self.weights])
            desired_goal = np.concatenate([desired_goal, self.weights])
        assert self.target.shape == (3,), f"target shape = {self.target.shape}"
        assert self.weights.shape == (3,), f"weights shape = {self.weights.shape}"
        # print(obs)
        if self.dictionary_obs:
            obs = {
        "observation": obs.astype(np.float32),
        "achieved_goal": achieved_goal.astype(np.float32),
        "desired_goal": desired_goal.astype(np.float32)
        }
        # print("NEW episode")
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

        if self.dictionary_obs:
            target = weights = np.array([])
        else:
            target = np.asarray(self.target).flatten()
            weights = np.asarray(self.weights).flatten()
            assert target.shape == (3,), f"Bad target shape {target.shape}"
            assert weights.shape == (3,), f"Bad weights shape {weights.shape}"

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
                combined = np.concatenate([obs[[0, 1, 4]] - target, weights, obs[[2, 3, 5]]])
            else:
                combined = np.concatenate([obs[[0, 1, 4]] - target, obs[[2, 3, 5]]])

        c = combined.astype(np.float32)
        return c
    
    def set_target_state(self, target):
        self.target = np.asarray(target).flatten()
    
    def set_weights(self, weights):
        self.weights = np.asarray(weights).flatten()

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Optional[dict[str, Any]]
    ) -> np.ndarray:
        # assert np.allclose(achieved_goal[3:6], desired_goal[3:6]), "weights mismatch"
        if self.weighted:
            if achieved_goal.ndim == 2:
                achieved = np.array(achieved_goal[:, :3], dtype=np.float32)
                weights = np.array(achieved_goal[:, 3:6], dtype=np.float32)
            else:
                achieved = np.array(achieved_goal[:3], dtype=np.float32)
                weights = np.array(achieved_goal[3:6], dtype=np.float32)
            diff = achieved
            diff = weights * diff
        else:
            diff = achieved_goal[:3]

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
                # print(result / np.max(result))
                return result / np.max(result)
    
    def target_state_fn(self, obs_space):
        # print("TW tragets")
        low = np.array([-1, -0.2, -2*np.pi])
        high = np.array([1, 1.5, 2*np.pi])
        return np.random.uniform(low, high)
    
class LunarLanderMPC(gym.Wrapper):
    def __init__(self, env : gym.Env):
        super().__init__(env)

        # Update observation space
        low = env.observation_space.low
        high = env.observation_space.high
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
        self.weights = np.array([1, 1, 1])
        self.target = np.array([0, 0, 0])
    
    @staticmethod
    def make_env():
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = LunarLanderMPC(env)
        return env
    
    def step(self, action):
        obs, reward, terminated, truncaded, info = self.env.step(action)
        obs = obs
        return obs, reward, terminated, truncaded, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return obs, info
        
    # Map Gym state to MPC state format
    def set_target_state(self, target_state):
        self.target = target_state
        self.env.set_target_state(target_state)

    def set_weights(self, weights):
        self.weights = weights
        self.env.set_weights(weights)