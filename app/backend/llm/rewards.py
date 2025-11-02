import re, json

def format_reward(output: str) -> float:
    tags = ["reasoning", "trajectory", "weights"]
    if not all(f"<{t}>" in output and f"</{t}>" in output for t in tags):
        return 0.0
    try:
        traj_text = re.search(r"<trajectory>(.*?)</trajectory>", output, re.S).group(1)
        json.loads(traj_text.replace("\n", "").strip())
        return 1.0
    except Exception:
        return 0.5  # tags ok but not parsable

import numpy as np

def bounds_reward(traj, weights, bounds_min, bounds_max):
    try:
        # Handle accidentally stringified bounds
        if isinstance(bounds_min, str):
            bounds_min = json.loads(bounds_min)
        if isinstance(bounds_max, str):
            bounds_max = json.loads(bounds_max)
        traj = np.asarray(traj, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        mins = np.asarray(bounds_min, dtype=np.float64)
        maxs = np.asarray(bounds_max, dtype=np.float64)

        # Check each element within bounds
        traj_in_bounds = np.all((traj >= mins) & (traj <= maxs))
        weights_in_bounds = np.all((weights >= 0) & (weights <= 1))

        # Hard binary reward
        if traj_in_bounds and weights_in_bounds:
            print("all good bounds")
            return 1.0
        else:
            print("bad bounds")
            return -1.0
    except:
        print("Error in bounds function, returning -1")
        print(f"traj: {traj}")
        print(f"weights: {weights}")
        print(f"bounds min: {bounds_min}")
        print(f"bounds max: {bounds_max}")
        return -1.0

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def trajectory_reward(expert, generated):
    try:
        distance, _ = fastdtw(expert, generated, dist=euclidean)
        d_norm = distance / (len(expert) + len(generated))
        return np.exp(-d_norm)
    except:
        print("Error in trajectory function, returning -1")
        print(f"expert: {expert}")
        print(f"generated: {generated}")
        return -1

def weights_reward(expert_w, generated_w):
    try:
        distance, _ = fastdtw(expert_w, generated_w, dist=euclidean)
        d_norm = distance / (len(expert_w) + len(generated_w))
        return np.exp(-0.5 * d_norm)
    except:
        print("Error in weights function, returning -1")
        print(f"expert_w: {expert_w}")
        print(f"generated_w: {expert_w}")
        return -1

def reasoning_reward(reasoning_text, states):
    count = sum(s.split()[0] in reasoning_text.lower() for s in states)
    if "weight" in reasoning_text.lower():
        count += 1
    return min(1.0, count / (len(states) + 1))

