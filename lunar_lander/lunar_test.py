import os
import gymnasium as gym
import custom_lunar_lander_no_target
from gymnasium.wrappers import RecordVideo
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.ppo.ppo import PPO
import numpy as np

def main():
    # First, create and train the environment without recording.
    train_env = gym.make("CustomLunarLander-v0", continuous=True)
    # model = PPO("MlpPolicy", train_env, verbose=1)
    model = PPO.load("models/last_model")
    model.set_env(train_env)
    model.learn(total_timesteps=500_000)
    train_env.close()  # Close training environment

    model.save("models/last_model")

    # Now, create a new environment for the final demonstration episode.
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)

    # For video recording, "render_mode" must be set to "rgb_array".
    demo_env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
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
