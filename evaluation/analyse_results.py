import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# 1. Global style
# --------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0

# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("tttttttt/combined")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OBS_SPACE_MAX = np.array([2, 1.5, 4 * np.pi])
START_STATE = np.array([0, 1.4, 0], dtype=np.float32)
THRESHOLDS = [0.1, 0.2, 0.5]

PPO_MODELS = [
    "PPO.dense.concat",
    "PPO.dense.diff",
    "PPO.sparse.concat",
    "PPO.sparse.diff",
]
SAC_MODELS = [
    "SAC.dense",
    "SAC.sparse",
]
MPC_MODELS = ["MPC.analytical", "MPC.SINDy", "OTR.SINDy"]
FAMILIES = {"PPO": PPO_MODELS, "SAC": SAC_MODELS, "Model Based": MPC_MODELS}

# --------------------------------------------------
# 3. Reference target distances
# --------------------------------------------------
targets_path = Path("datasets/targets.csv")
if not targets_path.exists():
    raise FileNotFoundError("Missing datasets/targets.csv")

targets = pd.read_csv(targets_path)[["x", "y", "theta"]]
targets["target_distance"] = np.linalg.norm(
    targets[["x", "y", "theta"]].values - START_STATE, axis=1
)
GLOBAL_MAX_DISTANCE = targets["target_distance"].max()

# --------------------------------------------------
# 4. Load and normalise results
# --------------------------------------------------
def load_results(model_name):
    path = RESULTS_DIR / f"{model_name}.csv"
    if not path.exists():
        print(f"⚠️ missing {path}")
        return None
    df = pd.read_csv(path)
    df["model"] = model_name
    df["target_distance"] = targets["target_distance"]
    for i, var in enumerate(["dx", "dy", "dtheta"]):
        df[f"{var}_norm"] = df[var] / OBS_SPACE_MAX[i]
    df["error_norm"] = np.linalg.norm(
        df[["dx_norm", "dy_norm", "dtheta_norm"]].values, axis=1
    )
    return df

# --------------------------------------------------
# 5. Load all data
# --------------------------------------------------
all_data = []
for fam, models in FAMILIES.items():
    for m in models:
        df = load_results(m)
        if df is not None:
            df["family"] = fam
            all_data.append(df)
data = pd.concat(all_data, ignore_index=True)
print(f"Loaded {len(data)} samples from {data['model'].nunique()} models.")

# --------------------------------------------------
# 6. Accuracy summary
# --------------------------------------------------
acc_rows = []
for model, mdf in data.groupby("model"):
    for thr in THRESHOLDS:
        acc = (mdf["error_norm"] < thr).mean() * 100
        acc_rows.append({"model": model, "threshold": thr, "accuracy(%)": acc})
acc_df = pd.DataFrame(acc_rows)
print("\nAccuracy summary (normalised error thresholds):")
print(acc_df.pivot(index="model", columns="threshold", values="accuracy(%)"))

# --------------------------------------------------
# 7. Determine best model per family
# --------------------------------------------------
best_per_family = {}
for fam, members in FAMILIES.items():
    fam_acc = acc_df[(acc_df["threshold"] == 0.2) & (acc_df["model"].isin(members))]
    if not fam_acc.empty:
        best_model = fam_acc.sort_values("accuracy(%)", ascending=False).iloc[0]["model"]
        best_per_family[fam] = best_model
print("\nBest models per family:", best_per_family)

# --------------------------------------------------
# 8. Plot utility: 3-stacked plots
# --------------------------------------------------
def plot_triplet(df, acc_subset, title, filename):
    """Create a 3-vertical-subplot figure comparing multiple models."""
    df["target_distance_norm"] = df["target_distance"] / GLOBAL_MAX_DISTANCE
    models = df["model"].unique()

    fig, axes = plt.subplots(3, 1, figsize=(7, 14), constrained_layout=True)
    palette = sns.color_palette("tab10", len(models))
    model_colors = dict(zip(models, palette))

    # 1️⃣ Error distribution — draw histograms manually per model
    for model in models:
        sub = df[df["model"] == model]
        sns.histplot(
            sub["error_norm"], bins=40, stat="count",
            color=model_colors[model], kde=True, label=model, ax=axes[0],
            alpha=0.4
        )
    for thr in THRESHOLDS:
        axes[0].axvline(thr, color="gray", ls="--", lw=1)
    axes[0].set_title(f"{title}: Error Distribution")
    axes[0].set_xlabel("Normalised error")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(left=0)
    axes[0].legend(title="Model", loc="upper right", frameon=False)

    # 2️⃣ Error vs Distance
    for model in models:
        sub = df[df["model"] == model]
        sns.scatterplot(
            data=sub, x="target_distance_norm", y="error_norm",
            color=model_colors[model], alpha=0.5, s=25, ax=axes[1], label=model
        )
        sns.regplot(
            data=sub, x="target_distance_norm", y="error_norm",
            scatter=False, color=model_colors[model],
            line_kws={"lw": 2, "ls": "--"}, ax=axes[1]
        )
    axes[1].set_title(f"{title}: Error vs Distance")
    axes[1].set_xlabel("Normalised target distance")
    axes[1].set_ylabel("Normalised error")
    axes[1].set_xlim(0, 1)
    axes[1].legend(title="Model", loc="upper right", frameon=False)

    # 3️⃣ Accuracy under thresholds (threshold legend)
    sns.barplot(
        data=acc_subset, x="model", y="accuracy(%)",
        hue="threshold", ax=axes[2], palette="Set2"
    )
    axes[2].set_title(f"{title}: Accuracy under Thresholds")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_ylim(bottom=0)
    axes[2].legend(title="Threshold", loc="upper right", frameon=False)
    axes[2].grid(True)

    # --- Save ---
    fig.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"✅ Saved {filename}")


# --------------------------------------------------
# 9. Generate combined triplet plots
# --------------------------------------------------
# 9.1 Best of each family
best_df = data[data["model"].isin(best_per_family.values())]
best_acc = acc_df[acc_df["model"].isin(best_per_family.values())]
plot_triplet(best_df, best_acc, "Best of Each Family", "best_of_each_family.png")

# 9.2 Each family: compare all models in that family
for fam, members in FAMILIES.items():
    fam_df = data[data["model"].isin(members)]
    fam_acc = acc_df[acc_df["model"].isin(members)]
    plot_triplet(fam_df, fam_acc, f"{fam} Family Comparison", f"{fam.lower()}_family_comparison.png")

# --------------------------------------------------
# 10. Summary stats
# --------------------------------------------------
print("\nAverage normalised error and variance per model:")
error_stats = (
    data.groupby("model")["error_norm"]
    .agg(["mean", "var"])
    .rename(columns={"mean": "mean_error", "var": "var_error"})
    .sort_values("mean_error")
)
for model, row in error_stats.iterrows():
    print(f"{model:35s}  mean={row['mean_error']:.4f}  var={row['var_error']:.4f}")

print(f"\nAll combined plots saved to {PLOTS_DIR}/")
