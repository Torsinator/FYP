import gymnasium as gym
import environments.custom_lunar_lander_no_target
from environments.custom_environment_wrapper import CustomEnvironmentWrapper
import numpy as np

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

def custom_lunar_lander():
    env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
    env = CustomEnvironmentWrapper(env, three_state_obs, three_state_reward, target_state_fn, weights_generation_fn)
    return env

envs = {"CustomLunarLander-v0" : custom_lunar_lander}

def get_env(env_name : str):
    return envs[env_name]()