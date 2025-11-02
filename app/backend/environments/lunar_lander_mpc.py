from typing import Any, Union, Optional
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
import environments.custom_lunar_lander_no_target


def target_state_fn(obs_space):
    low = np.array([-1, -0.3, -2*np.pi])
    high = np.array([1, 2.5, 2*np.pi])
    return np.random.uniform(0.9*low, 0.9*high)

def weights_generation_fn():
    while True:
        mask = np.random.random(3) < 0.5
        result = mask * np.random.random(3)
        if np.any(result != 0):
            print(result / np.max(result))
            return result / np.max(result)
        
def three_state_obs(obs, target_state, weights):
    pos_obs = obs[[0,1,4]]
    return np.concatenate((pos_obs, weights, obs[[2,3,5]]), dtype=np.float32)

class Lunar_Lander_MPC_Env(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.weights = None
        self.target_state = None
        self.min_distance = -np.inf

        # Update observation space
        low = np.concatenate((env.observation_space.low[[0,1,4]], [0,0,0], env.observation_space.low[[2,3,5]]))
        high = np.concatenate((env.observation_space.high[[0,1,4]], [1,1,1], env.observation_space.high[[2,3,5]]))
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": self.observation_space,                  # full state (Box(8,))
            "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        })
    
    @staticmethod
    def make_env():
        env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
        env = Lunar_Lander_MPC_Env(env)
        return env
    
    def step(self, action):
        assert(self.target_state is not None)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = three_state_obs(obs, self.target_state, self.weights)
        # if float(reward) > -1000:
        #     reward = self.reward_fn(obs, self.target_state, self.weights)
        if float(reward) < -1000:
            reward = -50
        else:
            reward = self.compute_reward(obs[[0,1,2]], self.target_state, info)
        return {
        "observation": obs.astype(np.float32),
        "achieved_goal": np.array(obs[[0,1,2]], dtype=np.float32),
        "desired_goal": np.array(self.target_state, dtype=np.float32)
    }, reward, reward == 1 or terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.min_distance = -np.inf
        if options is None:
            options = {}

        if "target_state" not in options:
            options["target_state"] = target_state_fn(self.env.observation_space)
        
        if "weights" not in options:
            options["weights"] = weights_generation_fn()

        self.target_state = options["target_state"]
        self.weights = options["weights"]
        print("TW weights:", self.weights)

        obs, info = super().reset(seed=seed, options=options)
        obs = three_state_obs(obs, self.target_state, self.weights)
        print("TW: ", self.target_state)
        return {
        "observation": obs.astype(np.float32),
        "achieved_goal": np.array(obs[[0,1,2]], dtype=np.float32),
        "desired_goal": np.array(self.target_state, dtype=np.float32)
    }, info
    
    def set_target_state(self, target_state):
        self.target_state = target_state
        self.env.unwrapped.set_target_state(target_state)
        print(f"target state set {target_state}")
    
    def set_weights(self, weights):
        self.weights = weights
        self.env.unwrapped.set_weights(weights)
        print(f"weights set {weights}")
    
    def set_reward_function(self, reward_fn):
        self.reward_fn = reward_fn

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, _info: Optional[dict[str, Any]]
    ) -> np.ndarray:
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)

        # difference (works for (3,) or (batch, 3))
        diff = achieved_goal - desired_goal

        # optional weighting/scaling
        # if self.weights is shape (3,), this will broadcast fine
        if hasattr(self, "weights"):
            diff = self.weights * diff

        dist = np.linalg.norm(diff, axis=-1)

        # sparse reward example: 0 if within tolerance, -1 otherwise
        return np.where(dist < 0.2, 1.0, -1.0).astype(np.float32)
        # return dist