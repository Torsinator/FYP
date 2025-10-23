import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import agents.PPO as PPO
import agents.SAC as SAC
from agents.SAC import *
from agents.PPO import *
from agents.SINDy import *
from agents.MPC import *

# Load the list of target goal states (x, y, theta)
targets = pd.read_csv("datasets/targets.csv")
weights_data = pd.read_csv("datasets/weights.csv")

# List all model classes you want to compare
sac_models = [
    # SAC_Unweighted_9_States_Sparse, SAC_Unweighted_9_States_Dense,
    # SAC_Unweighted_6_States_Sparse, SAC_Unweighted_6_States_Dense,
    # SAC_Weighted_9_States_Sparse, SAC_Weighted_9_States_Dense,
    # SAC_Weighted_6_States_Sparse, SAC_Weighted_6_States_Dense
]

ppo_models = [
    # PPO_Unweighted_9_States_Sparse, PPO_Unweighted_9_States_Dense,
    PPO_Unweighted_6_States_Sparse, PPO_Unweighted_6_States_Dense,
    # PPO_Weighted_9_States_Sparse, PPO_Weighted_9_States_Dense,
    # PPO_Weighted_6_States_Sparse, PPO_Weighted_6_States_Dense
]

# Combine for one loop
all_models = sac_models + ppo_models

# all_models = [MPC]

# Evaluate each model in turn
for model in all_models:
    print(f"\nRunning benchmark for: {model.__name__}")
    agent, env = model.load_model()
    results = []

    def calculate_error_6_state(target, current):
        actual_obs = target - current
        return actual_obs, np.linalg.norm(current), current

    def calculate_error(target, current):
        # print(target)
        # print(current)
        diff = target + current
        return np.linalg.norm(diff), diff

    # Evaluate each target state
    for i, target_state in tqdm(targets.iterrows()):
        weights = weights_data.iloc[i]
        target = np.array([target_state.x, target_state.y, target_state.theta], dtype=np.float32)
        # weights = np.array([weights.x, weights.y, weights.theta], dtype=np.float32)
        weights = np.array([1,1,1])

        # Reset environment with this goal
        obs, info = env.reset(seed=1, options={"target_state": target, "weights" : weights})
        if hasattr(env, 'target'):
            env.target = target
        # if hasattr(agent, 'set_target_state'):
        #     agent.set_target_state(target)
        # Record the starting position so we can measure how far the goal is
        obs_vec = obs.get("observation", obs) if isinstance(obs, dict) else obs
        start_state = np.array(obs_vec[[0, 1, 2]], dtype=np.float32)
        target_distance = np.linalg.norm(target - start_state)

        done = False
        best_error = np.inf
        best_diff = np.full(3, np.inf)

        # Run the episode until it ends
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, completed, terminated, info = env.step(action)
            done = completed or terminated

            obs_vec = obs.get("observation", obs) if isinstance(obs, dict) else obs
            if "6" in model.__name__:
                obs_vec, error, diff = calculate_error_6_state(target, obs_vec[[0, 1, 2]])
            else:
                error, diff = calculate_error(target, obs_vec[[0, 1, 2]])

            if error < best_error:
                best_error = error
                best_diff = diff
            if error < 0.1:
                print("close nuf")
                done = True

        results.append(np.concatenate([target, [best_error], best_diff, [target_distance]]))
        print(f"Target {i:03d}: error={best_error:.3f}, distance={target_distance:.3f}")

    # Store results in a DataFrame
    df = pd.DataFrame(results, columns=["tx", "ty", "ttheta", "error", "dx", "dy", "dtheta", "target_distance"])

    # Basic summary statistics
    mean_error = df["error"].mean()
    var_error = df["error"].var()
    success_rate = np.mean(df["error"] < 0.5) * 100

    print(f"\n{model.__name__} results:")
    print(f"  Mean error: {mean_error:.4f}")
    print(f"  Variance  : {var_error:.4f}")
    print(f"  Success (<0.5): {success_rate:.2f}%")

    # Save per-target results
    df["success"] = (df["error"] < 0.5).astype(int)
    df.to_csv(f"results/{model.__name__}.csv", index=False)

    # Plot 1: Error distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df["error"], bins=30, kde=True, color="steelblue")
    plt.title(f"{model.__name__}: Error distribution")
    plt.xlabel("Final tracking error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/{model.__name__}_error_dist.svg", dpi=300)
    plt.close()

    # Plot 2: Error vs target x, y, theta
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.scatterplot(ax=axes[0], x="tx", y="error", data=df, color="tab:blue")
    axes[0].set_title("Error vs Target X")
    sns.scatterplot(ax=axes[1], x="ty", y="error", data=df, color="tab:orange")
    axes[1].set_title("Error vs Target Y")
    sns.scatterplot(ax=axes[2], x="ttheta", y="error", data=df, color="tab:green")
    axes[2].set_title("Error vs Target θ")
    for ax in axes:
        ax.set_xlabel("Target value")
        ax.set_ylabel("Error")
    plt.suptitle(f"{model.__name__}: Correlation with target variables", y=1.05)
    plt.tight_layout()
    plt.savefig(f"plots/{model.__name__}_error_vs_targets.svg", dpi=300)
    plt.close()

    # Plot 3: Error vs distance from start
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="target_distance", y="error", data=df, color="purple")
    sns.regplot(x="target_distance", y="error", data=df, scatter=False, color="black", line_kws={"lw": 1, "ls": "--"})
    plt.title(f"{model.__name__}: Error vs distance from start")
    plt.xlabel("Distance between start and goal")
    plt.ylabel("Final error")
    plt.tight_layout()
    plt.savefig(f"plots/{model.__name__}_error_vs_distance.svg", dpi=300)
    plt.close()

    # Plot 4: Success vs failure rate
    plt.figure(figsize=(4, 4))
    sns.barplot(
        x=["Success", "Failure"],
        y=[df["success"].mean() * 100, (1 - df["success"].mean()) * 100],
        palette=["seagreen", "salmon"]
    )
    plt.title(f"{model.__name__}: Success rate")
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(f"plots/{model.__name__}_success_rate.svg", dpi=300)
    plt.close()

print("\nBenchmarking complete! Results saved to 'results/' and plots to 'plots/'.")
