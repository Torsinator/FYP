import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

def main():
    # First, create and train the environment without recording.
    train_env = gym.make("LunarLander-v3")
    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    train_env.close()  # Close training environment

    # Now, create a new environment for the final demonstration episode.
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)

    # For video recording, "render_mode" must be set to "rgb_array".
    demo_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    # Wrap the environment so that it always records the (only) episode.
    demo_env = RecordVideo(demo_env, video_folder=video_folder,
                           episode_trigger=lambda episode: True)

    # Run exactly one episode and record it.
    obs, info = demo_env.reset()
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
