import custom_lunar_lander_no_target

if __name__ == "__main__":
    import os
    import numpy as np
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    from stable_baselines3 import TD3
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from custom_environment_wrapper import CustomEnvironmentWrapper

    import matplotlib.pyplot as plt

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import plot_results
    from stable_baselines3.common import results_plotter
    from stable_baselines3.common.env_util import make_vec_env

    from datetime import datetime

    # Current timestamp as string
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create log directory
    log_dir = f"tmp-{timestamp_str}/"
    os.makedirs(log_dir, exist_ok=True)
    

    # rng = np.random.default_rng(seed=42)

    # --- Helper functions (same as before) ---
    def _normalise(x):
        return x / np.max(x)

    def three_state_reward(state, target_state, weights):
        distance = np.sqrt(np.sum(weights * (state[:3] - target_state[[0,1,4]]) ** 2))
        reward = -distance
        if distance < 0.01:
            reward += 10
        return reward

    def three_state_obs(obs, target_state, weights):
        pos_obs = obs[[0,1,4]]
        return np.concatenate((pos_obs, weights, obs[[2,3,5]]), dtype=np.float32)
        # return np.concatenate((pos_obs, obs[[2,3,5]]), dtype=np.float32)

    def target_state_fn(obs_space):
        low = np.array([-1, -0.2, -2*np.pi])
        high = np.array([1, 1.5, 2*np.pi])
        return np.random.uniform(low, high)

    def weights_generation_fn():
        while True:
            mask = np.random.random(3) < 0.5
            result = mask * np.random.random(3)
            if np.any(result != 0):
                print(result / np.max(result))
                return result / np.max(result)

    # --- Parallel environment ---
    def make_env(seed=None):
        def _init():
            env = gym.make("CustomLunarLander-v0", continuous=True)
            env = CustomEnvironmentWrapper(env, three_state_obs, three_state_reward,
                                           target_state_fn, weights_generation_fn)
            # if seed is not None:
            #     env.seed(seed)
            return env
        return _init()

    num_envs = 24
    # env = SubprocVecEnv([make_env(seed=i) for i in range(num_envs)])
    # env = Monitor(env, log_dir)
    env = make_vec_env(make_env, num_envs, monitor_dir=log_dir)

    # # --- HER + SAC ---
    goal_selection_strategy = GoalSelectionStrategy.FUTURE

    model = TD3(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=16,
            goal_selection_strategy=goal_selection_strategy,
        ),
        learning_starts=(num_envs+1)*50*20,
        verbose=1,
    )

    # model = SAC.load("./her_lunar_lander_model_weights_new", env=env)

    # # Train
    model.learn(2_000_000)
    model.save("./td3_her_2m")

    # Plot the results
    plot_results([log_dir], 2_000_000, results_plotter.X_TIMESTEPS, "TD3 with HER")
    plt.savefig(f"{log_dir}/plot")

    # --- Single demo video ---
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)

    demo_env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
    demo_env = CustomEnvironmentWrapper(demo_env, three_state_obs, three_state_reward,
                                        target_state_fn, weights_generation_fn)
    demo_env = RecordVideo(demo_env, video_folder=video_folder,
                           episode_trigger=lambda episode: True)

    obs, info = demo_env.reset(options={
        "target_state": np.array([0.5,0.5,3.14], dtype=np.float32),
        "weights": np.array([1,1,1], dtype=np.float32)
    })

    print("actual_obs", obs)
    print("obs_space", demo_env.observation_space)

    # model = SAC.load("./her_lunar_lander_model_weights_new", env=demo_env)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        done = terminated or truncated

    demo_env.close()
    print(f"Final demonstration video recorded and saved in: {video_folder}")
