import numpy as np
import re

def get_text_between_tags(text: str, tag: str) -> str:
        start_idx = text.find(f"<{tag}>")
        end_idx = text.find(f"</{tag}>")
        if start_idx >= end_idx:
            return ""
        return text[start_idx + len(tag) + 2:end_idx]

def parse_output(output : str, config):
    errors = []
    clarify = False
    traj = ""
    weights = ""
    reasoning = ""

    # Check case where we need clarification
    if "<clarification>" in output and "</clarification>" in output:
        clarify = True
        reasoning = get_text_between_tags(output, "clarification")
        
    else:
        reasoning = get_text_between_tags(output, "reasoning")
        traj = get_text_between_tags(output, "trajectory")
        weights = get_text_between_tags(output, "weights")

    return clarify, reasoning, traj, weights

    # --- Required sections ---
    # if "<reasoning>" not in output or "</reasoning>" not in output:
    #     errors.append("Missing <reasoning> tags.")
    # if "<trajectory>" not in output or "</trajectory>" not in output:
    #     errors.append("Missing <trajectory> tags.")
    # if "<weights>" not in output or "</weights>" not in output:
    #     errors.append("Missing <weights> tags.")

    # # --- Extract trajectory ---
    # traj_match = re.search(r"<trajectory>\s*(.*?)\s*</trajectory>", output, re.S)
    # weights_match = re.search(r"<weights>\s*(.*?)\s*</weights>", output, re.S)

    # if not traj_match or not weights_match:
    #     errors.append("Could not extract trajectory or weights.")
    #     return False

    # try:
    #     trajectory = np.array(eval(traj_match.group(1).strip()))
    #     weights = np.array(eval(weights_match.group(1).strip()))
    # except Exception as e:
    #     errors.append(f"Failed to parse trajectory/weights: {e}")
    #     return False

    # # --- Shape checks ---
    # if trajectory.ndim != 2:
    #     errors.append("Trajectory must be a 2D array.")
    # if weights.ndim != 2:
    #     errors.append("Weights must be a 2D array.")
    # if trajectory.shape != weights.shape:
    #     errors.append("Trajectory and weights must have same shape.")

    # # --- State dimension check ---
    # n_states = len(config["states"])
    # if trajectory.shape[1] != n_states:
    #     errors.append(f"Each trajectory row must have {n_states} states.")

    # # --- Bounds check ---
    # for i, (row, row_max, row_min) in enumerate(
    #     zip(trajectory, config["bounds"]["max"], config["bounds"]["min"])
    # ):
    #     if np.any(row > row_max) or np.any(row < row_min):
    #         errors.append(f"Trajectory row {i} has values outside bounds.")

    # # --- Weight range check ---
    # if not (np.all(weights >= 0) and np.all(weights <= 1)):
    #     errors.append("Weights must be between 0 and 1.")

    # # --- Report ---
    # if errors:
    #     print("Validation failed:")
    #     for e in errors:
    #         print(" -", e)
    #     return False
    # else:
    #     print("Output is valid")
    #     return True
