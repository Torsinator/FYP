import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)

def target_state_fn():
    # return rng.normal(loc=current_state[[0,1,4]], scale=0.2)
    low = np.array([-1, 0, -2*np.pi], dtype=np.float32)
    high = np.array([1, 1.5, 2*np.pi], dtype=np.float32)
    return rng.uniform(0.9*low, 0.9*high)

def _normalise(x):
    return x / np.sum(x)

def weights_generation_fn():
    # return _normalise(rng.uniform(0, 1, size=(6,)))
    while True:
        # 50% chance to be non-zero
        mask = rng.random(3) < 0.5  # True with probability `chance`
        # generate numbers in [0,1] where mask is True, else 0
        result = mask * rng.random(3)
        if np.any(result != 0):
            return _normalise(result)
        
def generate_targets(num):
    targets = []
    for i in range(num):
        targets.append(target_state_fn())

    column_names = ['x', 'y', 'theta']
    df_custom = pd.DataFrame(targets, columns=column_names)
    return df_custom

def generate_weights(num):
    weights = []
    for i in range(num):
        weights.append(weights_generation_fn())

    column_names = ['x', 'y', 'theta']
    df_custom = pd.DataFrame(weights, columns=column_names)
    return df_custom

def visualise_dataset_heatmap(dataset, title):
    """
    Visualize dataset as a scatter plot:
      - x, y as coordinates
      - theta as color (angle)
    """
    x = dataset["x"].to_numpy()
    y = dataset["y"].to_numpy()
    theta = dataset["theta"].to_numpy()

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        x, y,
        c=theta,
        cmap="hsv",        # cyclic colormap for angles
        s=60,
        edgecolors="k",
        alpha=0.8
    )
    plt.colorbar(sc, label="Angle (theta)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"datasets/{title}.svg")


def visualise_dataset_spikes(dataset):
    """
    Visualize dataset as three spike plots:
    - x-axis: sample index
    - y-axis: magnitude of each variable (x, y, theta)
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    columns = ["x", "y", "theta"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i, (ax, col, color) in enumerate(zip(axes, columns, colors)):
        values = dataset[col].to_numpy()
        indices = np.arange(len(values))

        # Plot as vertical spikes
        ax.vlines(indices, ymin=0, ymax=values, color=color, alpha=0.8, linewidth=1.5)
        ax.plot(indices, values, 'o', color=color, alpha=0.7, markersize=4)

        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle("Dataset Variable Magnitudes", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    targets = generate_targets(100)
    targets.to_csv("datasets/targets.csv", index=False)
    weights = generate_weights(100)
    weights.to_csv("datasets/weights.csv", index=False)
    visualise_dataset_heatmap(targets, "Target States (x, y, θ)")