import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# 1. Global style
# --------------------------------------------------
plt.style.use("seaborn-v0_8-paper")

# Remove all padding globally
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0

# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots/combined")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OBS_SPACE_MAX = np.array([2, 1.5, 4*np.pi])
START_STATE = np.array([0, 1.4, 0], dtype=np.float32)

THRESHOLDS = [0.1, 0.2, 0.5]

PPO_MODELS = [
    "PPO_Unweighted_9_States_Sparse",
    "PPO_Unweighted_9_States_Dense",
    "PPO_Unweighted_6_States_Sparse",
    "PPO_Unweighted_6_States_Dense",
]
SAC_MODELS = [
    "SAC_Unweighted_9_States_Sparse",
    "SAC_Unweighted_9_States_Dense",
]
MPC_MODELS = ["MPC", "SINDy"]
FAMILIES = {"PPO": PPO_MODELS, "SAC": SAC_MODELS, "MPC": MPC_MODELS}

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
# 4. Utility: load and normalise
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
# 7. Intra-family plots
# --------------------------------------------------
def plot_family(family_name, members):
    fam_df = data[data["model"].isin(members)]
    if fam_df.empty:
        print(f"No data for {family_name}")
        return

    # --- Error distribution ---
    plt.figure(figsize=(7,5))
    sns.histplot(data=fam_df, x="error_norm", hue="model", bins=40, kde=True, common_norm=False)
    for thr in THRESHOLDS:
        plt.axvline(thr, color="gray", ls="--", lw=1)
    plt.xlabel("Normalised error magnitude")
    plt.ylabel("Count")
    plt.title(f"{family_name} family: Error distribution")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.margins(0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/f"{family_name}_error_distribution.png", dpi=300)
    plt.close()

    # --- Error vs distance (normalised + per-model trendlines) ---
    plt.figure(figsize=(7,5))
    fam_df["target_distance_norm"] = fam_df["target_distance"] / GLOBAL_MAX_DISTANCE

    palette = sns.color_palette("tab10", len(members))
    model_colors = dict(zip(members, palette))

    sns.scatterplot(
        data=fam_df, x="target_distance_norm", y="error_norm",
        hue="model", palette=palette, alpha=0.5, s=25
    )
    for model, sub in fam_df.groupby("model"):
        sns.regplot(
            data=sub, x="target_distance_norm", y="error_norm",
            scatter=False, color=model_colors[model],
            line_kws={"lw":2, "ls":"--"}
        )

    plt.title(f"{family_name} family: Error vs Distance (per-model trends)")
    plt.xlabel("Normalised target distance")
    plt.ylabel("Normalised error")
    plt.xlim(0, 1.0)
    plt.ylim(bottom=0)
    plt.margins(0)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/f"{family_name}_error_vs_distance.png", dpi=300)
    plt.close()

    # --- Accuracy bars ---
    fam_acc = acc_df[acc_df["model"].isin(members)]
    plt.figure(figsize=(7,4))
    sns.barplot(data=fam_acc, x="model", y="accuracy(%)", hue="threshold")
    plt.title(f"{family_name} family: Accuracy under thresholds")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(bottom=0)
    plt.margins(0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/f"{family_name}_accuracy_bars.png", dpi=300)
    plt.close()

for fam, members in FAMILIES.items():
    plot_family(fam, members)

# --------------------------------------------------
# 8. Cross-family (best models)
# --------------------------------------------------
best_per_family = {}
for fam, members in FAMILIES.items():
    fam_acc = acc_df[(acc_df["threshold"] == 0.1) & (acc_df["model"].isin(members))]
    if not fam_acc.empty:
        best_model = fam_acc.sort_values("accuracy(%)", ascending=False).iloc[0]["model"]
        best_per_family[fam] = best_model
print("\nBest models per family:", best_per_family)

best_df = data[data["model"].isin(best_per_family.values())]
best_df["target_distance_norm"] = best_df["target_distance"] / GLOBAL_MAX_DISTANCE

# --- Combined error distribution ---
plt.figure(figsize=(7,5))
sns.histplot(data=best_df, x="error_norm", hue="model", bins=40, kde=True, common_norm=False)
for thr in THRESHOLDS:
    plt.axvline(thr, color="gray", ls="--", lw=1)
plt.xlabel("Normalised error")
plt.ylabel("Count")
plt.title("Best of each family: Error distribution")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.margins(0)
plt.tight_layout()
plt.savefig(PLOTS_DIR/"best_models_error_distribution.png", dpi=300)
plt.close()

# --- Error vs distance (normalised + per-model trendlines) ---
plt.figure(figsize=(7,5))
palette = sns.color_palette("tab10", len(best_df["model"].unique()))
model_colors = dict(zip(best_df["model"].unique(), palette))

sns.scatterplot(
    data=best_df, x="target_distance_norm", y="error_norm",
    hue="model", palette=palette, alpha=0.5, s=25
)
for model, sub in best_df.groupby("model"):
    sns.regplot(
        data=sub, x="target_distance_norm", y="error_norm",
        scatter=False, color=model_colors[model],
        line_kws={"lw":2, "ls":"--"}
    )

plt.xlabel("Normalised target distance")
plt.ylabel("Normalised error")
plt.title("Best of each family: Error vs Distance (per-model trends)")
plt.xlim(0, 1.0)
plt.ylim(bottom=0)
plt.margins(0)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(PLOTS_DIR/"best_models_error_vs_distance.png", dpi=300)
plt.close()

# --- Accuracy comparison bars ---
best_acc = acc_df[acc_df["model"].isin(best_per_family.values())]
plt.figure(figsize=(6,4))
sns.barplot(data=best_acc, x="model", y="accuracy(%)", hue="threshold", palette="magma")
plt.title("Best of each family: Accuracy under thresholds")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=20, ha="right")
plt.ylim(bottom=0)
plt.margins(0)
plt.tight_layout()
plt.savefig(PLOTS_DIR/"best_models_accuracy_bars.png", dpi=300)
plt.close()

# --------------------------------------------------
# Mean and variance of normalised error for each model
# --------------------------------------------------
print("\nAverage normalised error and variance per model:")
error_stats = (
    data.groupby("model")["error_norm"]
    .agg(["mean", "var"])
    .rename(columns={"mean": "mean_error", "var": "var_error"})
    .sort_values("mean_error")
)

# Pretty print table
for model, row in error_stats.iterrows():
    print(f"{model:35s}  mean={row['mean_error']:.4f}  var={row['var_error']:.4f}")

print(f"\nPlots saved to {PLOTS_DIR}/")
