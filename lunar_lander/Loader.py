import os
import gymnasium as gym
import custom_lunar_lander_no_target
from gymnasium.wrappers import RecordVideo
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.ppo.ppo import PPO
import numpy as np
from OpenAI_prompt import generate_state

def main():
    # First, create and train the environment without recording.
    demo_env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
    model = PPO.load("models/ppo_lunar_max_reward")
    model.set_env(demo_env)

    states = np.array(
  [
  [-0.70000, 0.8, 0, 0, 0, 0],
  [-0.65000, 0.5, 0, 0, 0, 0],
  [-0.60000, 0.2, 0, 0, 0, 0],
  [-0.55000, 0.5, 0, 0, 0, 0],
  [-0.50000, 0.8, 0, 0, 0, -0],
  [-0.45000, 0.5, 0, -0, -0, 0],
  [-0.40000, 0.2, 0, 0, 0, 0],
  [-0.35000, 0.5, 0, 0, 0, 0],
  [-0.30000, 0.8, 0, 0, 0, -0],
  [-0.25000, 0.5, 0, -0, -0, 0],
  [-0.20000, 0.2, 0, 0, 0, 0],
  [-0.15000, 0.5, 0, 0, 0, 0],
  [-0.10000, 0.8, 0, 0, 0, -0],
  [-0.05000, 0.5, 0, -0, -0, 0],
  [0.00000, 0.2, 0, 0, 0, 0],
  [0.05000, 0.5, 0, 0, 0, 0],
  [0.10000, 0.8, 0, 0, 0, -0],
  [0.15000, 0.5, 0, -0, -0, 0],
  [0.20000, 0.2, 0, 0, 0, 0],
  [0.25000, 0.5, 0, 0, 0, 0],
  [0.30000, 0.8, 0, 0, 0, -0],
  [0.35000, 0.5, 0, -0, -0, 0],
  [0.40000, 0.2, 0, 0, 0, 0],
  [0.45000, 0.5, 0, 0, 0, 0],
  [0.50000, 0.8, 0, 0, 0, -0],
  [0.55000, 0.5, 0, -0, -0, 0],
  [0.60000, 0.2, 0, 0, 0, 0],
  [0.65000, 0.5, 0, 0, 0, 0],
  [0.70000, 0.8, 0, 0, 0, -0]
  ]
    )

    n = len(states)  # number of copies
    row = np.array([1, 1, 0, 0, 0, 0])

    weights = np.tile(row, (n, 1))

    # Now, create a new environment for the final demonstration episode.
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)

    # Wrap the environment so that it always records the (only) episode.
    demo_env = RecordVideo(demo_env, video_folder=video_folder,
                           episode_trigger=lambda episode: True)

    # Run exactly one episode and record it.
    obs, info = demo_env.reset(options={"target_state" : states[0], "weights": weights[0]})
    # obs, info = demo_env.reset(options={"target_state" : generate_state("Spin at 0.1 rad/s at (0, 0.5)")})
    # done = False
    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = demo_env.step(action)
    #     done = terminated or truncated

    # Finalize recording by closing the environment.
    # demo_env.close()
    # print(f"Final demonstration video recorded and saved in: {video_folder}")

    for target_state in states:
        done = False
        demo_env.unwrapped.set_target_state(np.array(target_state, dtype=np.float32))
        demo_env.unwrapped.set_weights(weights[0])
        # === Control Loop ===
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = demo_env.step(action)
            done = terminated or truncated or abs(reward) > 999

    demo_env.close()
    print(f"Final demonstration video recorded and saved in: {video_folder}")

if __name__ == '__main__':
    main()
