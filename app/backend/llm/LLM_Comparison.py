import unsloth
import torch, json, re, gc
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import GenerationConfig
from unsloth import FastLanguageModel
import rewards  # your rewards.py

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_BASE = "unsloth/gpt-oss-20b"
MODEL_FINETUNED = "planning_model"
DATA_PATH = "lunar_dataset.json"
N_SAMPLES = 10
MAX_TOKENS = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# GENERATION SETTINGS
# ============================================================
gen_cfg = GenerationConfig(
    max_new_tokens=4096,
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    top_k=0
)

# ============================================================
# HELPERS
# ============================================================
def extract_final_output(text: str) -> str:
    """
    Extracts only the assistant's final output — everything from the LAST
    <reasoning> (or <clarification>) tag onward.
    """
    if not text or not isinstance(text, str):
        return ""

    # Search for the last occurrence of <reasoning> or <clarification>
    match = None
    for tag in ["<reasoning>", "<clarification>"]:
        m = list(re.finditer(tag, text))
        if m:
            # keep the latest match overall
            if not match or m[-1].start() > match.start():
                match = m[-1]

    if match:
        final = text[match.start():].strip()
        return final

    # fallback: nothing found, return stripped text
    return text.strip()

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=gen_cfg)
    return extract_final_output(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

def extract_block(text, tag):
    m = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else ""

import ast

def safe_json(text: str):
    """Safely parse a JSON- or Python-style list, handling whitespace/newlines."""
    if not text or not isinstance(text, str):
        return []
    try:
        # Clean up formatting from model output
        return ast.literal_eval(text)
    except Exception as e:
        print(f"[safe_json] Parse error: {e}\n{text=}")
        return []


def compute_scores(output, expert, states, bounds_min, bounds_max):
    """Compute reward subcomponents and total."""
    try:
        reasoning = extract_block(output, "reasoning")
        traj = safe_json(extract_block(output, "trajectory"))
        w = safe_json(extract_block(output, "weights"))
        r = {
            "format": rewards.format_reward(output),
            "bounds": rewards.bounds_reward(traj, w, bounds_min, bounds_max),
            "trajectory": rewards.trajectory_reward(expert["trajectory"], traj),
            "weights": rewards.weights_reward(expert["weights"], w),
            "reasoning": rewards.reasoning_reward(reasoning, states),
        }
        r["total"] = (
            0.3*r["format"] + 0.3*r["bounds"] +
            0.2*r["trajectory"] + 0.1*r["weights"] + 0.1*r["reasoning"]
        )
    except Exception as e:
        print(f"[Warn] Reward error: {e}")
        r = {k: -1 for k in ["format","bounds","trajectory","weights","reasoning","total"]}
    return r

# ============================================================
# DATASET
# ============================================================
dataset = load_dataset("json", data_files=DATA_PATH, split=f"train[:{N_SAMPLES}]")

# ============================================================
# RUN EVALUATION FUNCTION
# ============================================================

def generate_system_prompt(context, states, max, min, current_state):
    return f"""
{context}

TASK
Generate a minimal, sufficient **trajectory** of target states and a matching 2D array of per-variable **weights** for the given command.

DEFINITIONS
States: {states}
Bounds: min = {min}, max = {max}
Current state: {current_state}

STRICT OUTPUT SPEC (MUST FOLLOW EXACTLY)
- If the instruction is ambiguous or any required info is missing, output **only**:
  <clarification>clear_text_explaining_what_is_missing_or_ambiguous</clarification>
  (No other text allowed.)

- Otherwise output **exactly these three tags in this order** and nothing else:
  1) <reasoning>...</reasoning>
     - Free text explaining assumptions and why each target state & weighting was chosen.
     - How this trajectory meets the user's request
     - Keep it concise (max ~6 short sentences).
  2) <trajectory>[[...],[...],...]</trajectory>
     - A 2D JSON array (list of rows) of numeric **floats** only.
     - Each row = one target state; each column corresponds to the state variables listed above.
     - Use decimal notation (e.g. 0.125 or 1.0). **Do not** use scientific notation (`1e-3`), expressions, variable names, comments, or trailing commas.
  3) <weights>[[...],[...],...]</weights>
     - A 2D JSON array of floats with **exactly the same shape** as `<trajectory>`.
     - Every element must be in range [0.0, 1.0].
     - Weights are the importance per state variable in the target state not the state itself. These can change at different target states.
     - No extra text or formatting.

VALIDATION STEPS (you must perform these checks before returning)
1. Shape: number of columns in each `<trajectory>` row == number of state variables in `States`. Number of rows in `<weights>` == number of rows in `<trajectory>`. Each corresponding row length must match.
2. Bounds: every trajectory value must satisfy `min <= value <= max` for the corresponding state variable.
3. Weights: every weight must satisfy `0.0 <= weight <= 1.0`.
4. Formatting: `<trajectory>` and `<weights>` must be valid JSON arrays containing only numeric literals (no comments, no text).
If any check fails, **do not** output reasoning or arrays — output **only** a `<clarification>` tag listing the failing checks (short, comma-separated).

ADDITIONAL RULES
- Do not include any other tags or text outside the tags described above.
- Do not guess: if you must assume something to proceed, stop and request clarification using `<clarification>`.
- Include the current state as the first entry in the trajectory

EXAMPLE (format only — replace with real numbers that respect bounds and shapes)
<reasoning>Concise reason for states and weights.</reasoning>
<trajectory>[[0.0, 1.0, 0.5], [0.2, 0.9, 0.1]]</trajectory>
<weights>[[1.0, 0.8, 0.2], [0.9, 0.7, 0.1]]</weights>

Now produce the output for the command that follows.
"""

def format_output(reasoning, trajectory, weights):
    return f'''<reasoning>{reasoning}</reasoning>
<trajectory>{trajectory}</trajectory>
<weights>{weights}</weights>
'''

def evaluate_model(model_name, tag):
    print(f"\n=== Loading {tag} model: {model_name} ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        max_seq_length=MAX_TOKENS
    )

    results = []
    for i, ex in enumerate(dataset):
        system = ex["context"]["system"]
        instruction = ex["instruction"]
        states = ex["context"]["states"]
        bounds_min, bounds_max = ex["context"]["bounds"]["min"], ex["context"]["bounds"]["max"]
        expert = ex["output"]
        trajectory = expert["trajectory"]
        instr = generate_system_prompt(system, states, bounds_max, bounds_min, trajectory[0])

        convo = [
            {"role": "system", "content": instr},
            {"role": "user", "content": instruction},
        ]
        prompt = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True, reasoning_effort="low")
        output = generate(model, tokenizer, prompt)
        print(output)

        score = compute_scores(output, expert, states, bounds_min, bounds_max)
        results.append({"id": i+1, "instruction": instruction, "system": system, **score})
        print(f"→ [{tag}] task {i+1}/{N_SAMPLES}: total={score['total']:.3f}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results

# ============================================================
# SEQUENTIAL EVALUATION
# ============================================================
fine_results = evaluate_model(MODEL_FINETUNED, "Fine-tuned")
base_results = evaluate_model(MODEL_BASE, "Base")
# ============================================================
# COMBINE RESULTS
# ============================================================
df_base = pd.DataFrame(base_results)
df_fine = pd.DataFrame(fine_results)

# --- Merge sub-reward metrics for comparison ---
reward_metrics = ["format", "bounds", "trajectory", "weights", "reasoning", "total"]

# Merge all into one DataFrame for clarity
merged = pd.DataFrame({
    "id": df_base["id"],
    "instruction": df_base["instruction"],
    "system": df_base["system"],
})
for m in reward_metrics:
    merged[f"base_{m}"] = df_base[m]
    merged[f"fine_{m}"] = df_fine[m]
    merged[f"Δ_{m}"] = merged[f"fine_{m}"] - merged[f"base_{m}"]

# ============================================================
# TABLE 1 — MEAN SCORES PER METRIC
# ============================================================
summary = pd.DataFrame({
    "Metric": reward_metrics,
    "Base Mean": [df_base[m].mean() for m in reward_metrics],
    "Fine Mean": [df_fine[m].mean() for m in reward_metrics],
})
summary["Δ Improvement"] = summary["Fine Mean"] - summary["Base Mean"]

print("\n" + "="*70)
print("TABLE 1 — MEAN SCORES PER REWARD METRIC")
print("="*70)
print(summary.to_markdown(index=False, floatfmt=".3f"))

# ============================================================
# TABLE 2 — PER-TASK TOTAL AND SUBREWARDS
# ============================================================
task_cols = (
    ["id", "instruction", "system"]
    + [f"base_{m}" for m in reward_metrics]
    + [f"fine_{m}" for m in reward_metrics]
    + [f"Δ_{m}" for m in reward_metrics]
)
task_table = merged[task_cols]

print("\n" + "="*70)
print("TABLE 2 — DETAILED PER-TASK REWARD COMPARISON")
print("="*70)
print(task_table.to_markdown(index=False, floatfmt=".3f"))

# ============================================================
# SAVE OUTPUTS
# ============================================================
summary.to_csv("mean_scores_summary.csv", index=False)
task_table.to_csv("task_scores_detailed.csv", index=False)
print("\nSaved:")
print("  • mean_scores_summary.csv")
print("  • task_scores_detailed.csv")
