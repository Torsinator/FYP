import os
import gymnasium as gym
import custom_lunar_lander_no_target
from gymnasium.wrappers import RecordVideo
from stable_baselines3.ppo.ppo import PPO
import numpy as np
from scipy.interpolate import interp1d

EPISODE_LENGTH_SECONDS = 40
HZ = 50
DT = 1.0 / HZ


def interpolate_states(states, episode_length, hz):
    """
    Interpolate keyframe states across the episode timeline.
    states: (N, state_dim) keyframes
    Returns: (T, state_dim) interpolated trajectory
    """
    num_waypoints = len(states)
    state_dim = states.shape[1]

    # spread target state across full episode time
    t_waypoints = np.linspace(0, episode_length, num_waypoints)
    # timeline at 50Hz
    t_dense = np.linspace(0, episode_length, int(episode_length * hz))

    states_interp = np.zeros((len(t_dense), state_dim))
    for d in range(state_dim):
        f = interp1d(t_waypoints, states[:, d], kind="linear")
        states_interp[:, d] = f(t_dense)

    return states_interp


def main():
    demo_env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
    model = PPO.load("models/last_model")
    model.set_env(demo_env)

    # Your original keyframes
    states = np.array([
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

    # Include lander starting position as extra first state
    start_state = np.array([0, 1.4, 0, 0, 0, 0])
    states = np.vstack([start_state, states])

    # Interpolate trajectory
    states_interp = interpolate_states(states, EPISODE_LENGTH_SECONDS, HZ)

    # Record video
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)
    demo_env = RecordVideo(demo_env, video_folder=video_folder, episode_trigger=lambda episode: True)

    obs, info = demo_env.reset(options={"target_state": states_interp[0]})

    # Step through the interpolated trajectory
    for target_state in states_interp:
        demo_env.unwrapped.set_target_state(np.array(target_state, dtype=np.float32))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        # if terminated or truncated:
        #    break

    demo_env.close()
    print(f"Final demonstration video recorded and saved in: {video_folder}")


if __name__ == '__main__':
    main()
