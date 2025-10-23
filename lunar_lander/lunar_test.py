import os
import gymnasium as gym
import custom_lunar_lander_no_target
from gymnasium.wrappers import RecordVideo
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.ppo.ppo import PPO
from sb3_contrib import RecurrentPPO
from custom_environment_wrapper import CustomEnvironmentWrapper
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
    pos_ts = target_state
    # print((target_state - obs) * weights)
    # return (target_state - obs) * weights
    return np.concatenate((pos_obs, pos_ts, obs[[2,3,5]]))

def target_state_fn(obs_space):
    # return rng.normal(loc=current_state[[0,1,4]], scale=0.2)
    low = np.array([-2.5, 0, -2*np.pi])
    high = np.array([2.5, 2.5, 2*np.pi])
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

def main():
    # First, create and train the environment without recording.
    train_env = gym.make("CustomLunarLander-v0", continuous=True)
    train_env = CustomEnvironmentWrapper(train_env, three_state_obs, three_state_reward, target_state_fn, weights_generation_fn)
    model = PPO("MlpPolicy", train_env, verbose=1, device="cpu")
    # model = PPO.load("models/last_model", device="cpu")
    model.set_env(train_env)
    model.learn(total_timesteps=2_000_000)
    train_env.close()  # Close training environment

    model.save("models/last_model")

    # Now, create a new environment for the final demonstration episode.
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)

    # For video recording, "render_mode" must be set to "rgb_array".
    demo_env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
    demo_env = CustomEnvironmentWrapper(demo_env, three_state_obs, three_state_reward, target_state_fn, weights_generation_fn)
    # Wrap the environment so that it always records the (only) episode.
    demo_env = RecordVideo(demo_env, video_folder=video_folder,
                           episode_trigger=lambda episode: True)

    # Run exactly one episode and record it.
    obs, info = demo_env.reset(options={"target_state" : np.array([0,0,0,0,0,0], dtype=np.float32)})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        done = terminated or truncated

    # Finalize recording by closing the environment.
    demo_env.close()
    print(f"Final demonstration video recorded and saved in: {video_folder}")

if __name__ == '__main__':
    main()
