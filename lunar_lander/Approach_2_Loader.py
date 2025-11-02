import os
import gymnasium as gym
import custom_lunar_lander_no_target
from gymnasium.wrappers import RecordVideo
from stable_baselines3.ppo.ppo import PPO
import numpy as np
from OpenAI_prompt import generate_state
from scipy.interpolate import interp1d
from custom_environment_wrapper import CustomEnvironmentWrapper

EPISODE_LENGTH_SECONDS = 20  # <-- set this
HZ = 50
DT = 1.0 / HZ

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
    print(len(weights))
    print(len(np.concatenate((pos_ts - pos_obs, weights, obs[[2,3,5]]))))
    return np.concatenate((pos_ts - pos_obs, weights, obs[[2,3,5]]))

def target_state_fn(obs_space):
    # return rng.normal(loc=current_state[[0,1,4]], scale=0.2)
    low = np.array([-2.5, 0, -2*np.pi, 0, 0, 0])
    high = np.array([2.5, 2.5, 2*np.pi, 1, 1, 1])
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

def interpolate_states(states, episode_length, hz):
    """
    Interpolate target states so that we have one target per timestep.
    states: array [N, state_dim]
    """
    num_waypoints = len(states)
    state_dim = states.shape[1]

    # Original timepoints (spread across episode)
    t_waypoints = np.linspace(0, episode_length, num_waypoints)

    # New dense timeline at desired resolution
    t_dense = np.linspace(0, episode_length, int(episode_length * hz))

    # Interpolate each dimension separately
    states_interp = np.zeros((len(t_dense), state_dim))
    for d in range(state_dim):
        f = interp1d(t_waypoints, states[:, d], kind="linear")
        states_interp[:, d] = f(t_dense)

    return states_interp


def main():
    demo_env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
    demo_env = CustomEnvironmentWrapper(demo_env, three_state_obs, three_state_reward, target_state_fn, weights_generation_fn)
    model = PPO.load("models/last_model")
    model.set_env(demo_env)

    # Keyframes / waypoints
    states = np.array([
        [0, 1.4, 0, 0, 0, 0],
        [-0.70000, 0.8, 0, 0, 0, 0],
        [-0.65000, 0.5, 0, 0, 0, 0],
        [-0.60000, 0.2, 0, 0, 0, 0],
        [-0.55000, 0.5, 0, 0, 0, 0],
        [-0.50000, 0.8, 0, 0, 0, 0],
        [-0.45000, 0.5, 0, 0, 0, 0],
        [-0.40000, 0.2, 0, 0, 0, 0],
        [-0.35000, 0.5, 0, 0, 0, 0],
        [-0.30000, 0.8, 0, 0, 0, 0],
        [-0.25000, 0.5, 0, 0, 0, 0],
        [-0.20000, 0.2, 0, 0, 0, 0],
        [-0.15000, 0.5, 0, 0, 0, 0],
        [-0.10000, 0.8, 0, 0, 0, 0],
        [-0.05000, 0.5, 0, 0, 0, 0],
        [0.00000, 0.2, 0, 0, 0, 0],
        [0.05000, 0.5, 0, 0, 0, 0],
        [0.10000, 0.8, 0, 0, 0, 0],
        [0.15000, 0.5, 0, 0, 0, 0],
        [0.20000, 0.2, 0, 0, 0, 0],
        [0.25000, 0.5, 0, 0, 0, 0],
        [0.30000, 0.8, 0, 0, 0, 0],
        [0.35000, 0.5, 0, 0, 0, 0],
        [0.40000, 0.2, 0, 0, 0, 0],
        [0.45000, 0.5, 0, 0, 0, 0],
        [0.50000, 0.8, 0, 0, 0, 0],
        [0.55000, 0.5, 0, 0, 0, 0],
        [0.60000, 0.2, 0, 0, 0, 0],
        [0.65000, 0.5, 0, 0, 0, 0],
        [0.70000, 0.8, 0, 0, 0, 0],
    ])

    n = len(states)  # number of copies
    # row = np.array([0, 0, 1, 1, 1, 1])
    row = np.array([1, 1, 0, 0, 0, 0])

    weights = np.tile(row[[0,1,4]], (n, 1))

    # Interpolate to per-timestep targets
    states_interp = interpolate_states(states, EPISODE_LENGTH_SECONDS, HZ)

    weights_interp = interpolate_states(weights, EPISODE_LENGTH_SECONDS, HZ)

    print("TW", len(weights_interp))

    # Record video
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)
    demo_env = RecordVideo(demo_env, video_folder=video_folder, episode_trigger=lambda episode: True)

    obs, info = demo_env.reset(options={"target_state": states_interp[0], "weights": weights_interp[0]})

    # Step through dense interpolated targets
    for i in range(len(states_interp)):
        done = False
        demo_env.unwrapped.set_target_state(np.array(states_interp[i], dtype=np.float32))
        print(weights_interp[i])
        demo_env.unwrapped.set_weights(np.array(weights_interp[i], dtype=np.float32))
        # while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        done = terminated or truncated

    demo_env.close()
    print(f"Final demonstration video recorded and saved in: {video_folder}")


if __name__ == '__main__':
    main()
