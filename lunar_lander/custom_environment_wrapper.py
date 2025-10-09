from typing import Any, Union, Optional
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
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
        self.min_distance = -np.inf
        self.last_dist = None

        # Update observation space
        low = np.concatenate((env.observation_space.low[[0,1,4]], env.observation_space.low[[0,1,4]], env.observation_space.low[[2,3,5]]))
        high = np.concatenate((env.observation_space.high[[0,1,4]], env.observation_space.high[[0,1,4]], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        # self.observation_space = spaces.Dict({
        #     "observation": self.observation_space,                  # full state (Box(8,))
        #     "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        #     "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        # })
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.obs_fn(obs, self.target_state, self.weights)
        # if float(reward) > -1000:
        #     reward = self.reward_fn(obs, self.target_state, self.weights)
        reward = self.compute_reward(obs[[0,1,2]], self.target_state, info)
    #     return {
    #     "observation": obs.astype(np.float32),
    #     "achieved_goal": np.array(obs[[0,1,2]], dtype=np.float32),
    #     "desired_goal": np.array(self.target_state, dtype=np.float32)
    # }, reward, reward == 1 or terminated, truncated, info
        return obs, reward, reward == 100 or terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.min_distance = -np.inf
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
        self.last_dist = np.linalg.norm(obs[[0,1,2]] - self.target_state, axis=-1)
        print("TW: ", self.target_state)
    #     return {
    #     "observation": obs.astype(np.float32),
    #     "achieved_goal": np.array(obs[[0,1,2]], dtype=np.float32),
    #     "desired_goal": np.array(self.target_state, dtype=np.float32)
    # }, info
        return obs, info
    
    def set_target_state(self, target_state):
        self.target_state = target_state
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_reward_function(self, reward_fn):
        self.reward_fn = reward_fn
    
    def set_obs_function(self, obs_fn):
        self.obs_fn = obs_fn

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, _info: Optional[dict[str, Any]]
    ) -> np.ndarray:
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)

        # difference (works for (3,) or (batch, 3))
        diff = achieved_goal - desired_goal
        diff = np.linalg.norm(diff, axis=-1)

        # optional weighting/scaling
        # if self.weights is shape (3,), this will broadcast fine
        # if hasattr(self, "weights"):
        #     diff = self.weights * diff

        dist = self.last_dist - diff

        self.last_dist = diff

        # sparse reward example: 0 if within tolerance, -1 otherwise
        if diff > 0.1:
            return dist
        else:
            return np.array(100.0)
        # return dist
        # return dist