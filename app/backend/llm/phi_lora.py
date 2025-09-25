"""
llm_tagged_answer_planner.py

Requirements:
  pip install transformers accelerate bitsandbytes torch

Example usage: python llm_tagged_answer_planner.py
"""

import re
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Schema helpers
# -------------------------
@dataclass
class VarSpec:
    name: str
    min: float
    max: float
    unit: Optional[str] = None

@dataclass
class StateSchema:
    vars: List[VarSpec]
    @property
    def dim(self): return len(self.vars)
    @property
    def names(self): return [v.name for v in self.vars]
    @property
    def mins(self): return [v.min for v in self.vars]
    @property
    def maxs(self): return [v.max for v in self.vars]

# -------------------------
# Robust extraction & parsing for <answer> tags
# -------------------------
TAG_RE = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE | re.MULTILINE)

def extract_answer_block(text: str) -> str:
    """
    Return the content inside the first <answer>...</answer> block.
    If not found, raise ValueError.
    """
    m = TAG_RE.search(text)
    if not m:
        raise ValueError("No <answer>...</answer> block found in model output.")
    return m.group(1).strip()

def parse_plan(text: str) -> Dict[str, Any]:
    """
    Try to parse text as JSON. If fails, apply some light fixes then parse.
    Accepted final formats:
      - {"weights": [...], "trajectory": [[...], ...]}
      - {"trajectory": [[...], ...]}
      - [[...], ...]   (just a list of rows)
    Returns dict with keys 'weights' (or None) and 'trajectory' (list).
    """
    print(text)
    # direct JSON attempt
    try:
        obj = json.loads(text)
    except Exception:
        # easy repairs: remove trailing commas, convert single-quotes to double, remove stray words
        repaired = text.replace("\n"," ").replace(",]", "]").replace(",}", "}")
        repaired = repaired.replace("'", '"')
        # try to find a JSON array or object
        json_like = None
        # try to find { ... "trajectory": [...] ... }
        traj_re = re.compile(r'\{.*"trajectory"\s*:\s*\[.*\].*\}', re.I)
        m = traj_re.search(repaired)
        if m:
            json_like = m.group(0)
        else:
            arr_re = re.compile(r'\[\s*\[.*\]\s*(,\s*\[.*\]\s*)*\]', re.S)
            m2 = arr_re.search(repaired)
            if m2:
                json_like = m2.group(0)
        if json_like is None:
            # last resort: strip non-json characters and attempt load
            json_like = repaired
        try:
            obj = json.loads(json_like)
        except Exception as e:
            raise ValueError(f"Could not parse answer block as JSON-like data. Error: {e}\nBlock:\n{text}")
    # normalize
    result = {}
    if isinstance(obj, list):
        result["trajectory"] = obj
        result["weights"] = None
    elif isinstance(obj, dict):
        traj = obj.get("trajectory", None)
        w = obj.get("weights", None)
        if traj is None:
            # maybe the object itself is the array under a different key; fall back
            # find the first array value
            for v in obj.values():
                if isinstance(v, list):
                    traj = v
                    break
        if traj is None:
            raise ValueError("Parsed JSON object but could not find a trajectory array.")
        result["trajectory"] = traj
        result["weights"] = w
    else:
        raise ValueError("Parsed JSON is neither list nor dict.")
    return result

# -------------------------
# Projection: clamp + simple smoothing
# -------------------------
def clamp_and_smooth(traj: List[List[float]],
                     schema: StateSchema,
                     max_delta: Optional[List[float]] = None,
                     smooth_alpha: float = 0.12) -> List[List[float]]:
    T = len(traj)
    V = schema.dim
    mins = schema.mins
    maxs = schema.maxs

    # Ensure every row has V elements (pad with last value or zeros)
    fixed = []
    for row in traj:
        if not isinstance(row, list):
            raise ValueError("Trajectory rows must be lists of numbers.")
        r = [float(x) for x in row[:V]] + [0.0] * max(0, V - len(row))
        fixed.append(r)

    # clamp
    for t in range(T):
        for v in range(V):
            fixed[t][v] = max(mins[v], min(maxs[v], fixed[t][v]))

    # limit per-step delta if provided
    if max_delta is not None:
        for t in range(1, T):
            for v in range(V):
                d = fixed[t][v] - fixed[t-1][v]
                cap = max_delta[v] if v < len(max_delta) else None
                if cap is not None:
                    if d > cap: fixed[t][v] = fixed[t-1][v] + cap
                    if d < -cap: fixed[t][v] = fixed[t-1][v] - cap

    # exponential smoothing
    if smooth_alpha > 0 and T > 1:
        sm = fixed[0].copy()
        out = [sm.copy()]
        for t in range(1, T):
            for v in range(V):
                sm[v] = (1 - smooth_alpha) * sm[v] + smooth_alpha * fixed[t][v]
            out.append(sm.copy())
        return out
    return fixed

# -------------------------
# LLM wrapper (tagged-answer)
# -------------------------
class TagAnswerPlanner:
    def __init__(self, model_name: str = "microsoft/phi-3-mini-4k-instruct",
                 device: str = "cuda", load_4bit: bool = True):
        kwargs = {}
        # if load_4bit:
        #     kwargs.update(dict(load_in_4bit=True, device_map="auto", torch_dtype=torch.float16))
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device).eval()
        self.device = device

    def build_prompt(self, instruction: str, schema: StateSchema, allow_chain_of_thought=True) -> str:
        # give schema and allow chain of thought but force final answer in <answer> tags
        schema_lines = []
        for i, v in enumerate(schema.vars):
            rng = f"[{v.min}, {v.max}]"
            unit = f" {v.unit}" if v.unit else ""
            schema_lines.append(f"- {i}: {v.name}{unit} in {rng}")
        schema_txt = "\n".join(schema_lines)
        preface = ""
        if allow_chain_of_thought:
            preface = (
                "You may think step-by-step or explain your reasoning. "
                "However, the FINAL machine-readable plan MUST appear only inside the <answer>...</answer> block, "
                "and nothing outside those tags will be parsed by the system. "
                "Do not put the plan anywhere else.\n\n"
            )
        fmt = (
            "The plan format inside <answer> must be strict JSON with keys 'weights' and 'trajectory'.\n"
            "Example:\n"
            "{\n  \"weights\": [ [w0_0, w0_1, ...], [w1_0, w1_1, ...], ... ],\n  \"trajectory\": [ [s0_0, s0_1, ...], [s1_0, s1_1, ...], ... ]\n}\n"
            "Weights is a rating of the importance of each state variable for each state in the trajectory. The instruction may not always prescribe specific targets.\n"
            "Trajectory must contain the important target states to capture the behaviour of the prompt. Length of weights must equal length of trajectory.\n"
        )
        prompt = (
            preface +
            f"Instruction:\n{instruction}\n\n"
            f"State schema (index: name [min,max]):\n{schema_txt}\n\n"
            + fmt +
            "Remember: You should reason, but only the content inside <answer>...</answer> will be used.\n\n"
            "<answer>\n"
        )
        return prompt

    def generate_plan(self,
                      instruction: str,
                      schema: StateSchema,
                      temperature: float = 0.2,
                      top_p: float = 0.9,
                      max_new_tokens: int = 1024) -> Dict[str, Any]:
        prompt_prefix = self.build_prompt(instruction, schema, allow_chain_of_thought=True)
        prompt_suffix = "\n</answer>\n"  # close tag included in prompt to guide model to fill inside
        full_prompt = prompt_prefix + prompt_suffix
        inputs = self.tok(full_prompt, return_tensors="pt").to(self.model.device)

        # generate
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tok.eos_token_id,
            )
        print(out)
        txt = self.tok.decode(out[0], skip_special_tokens=True)
        # Extract answer block
        try:
            print("TW text: ", txt)
            block = extract_answer_block(txt)
        except Exception as e:
            raise RuntimeError(f"Failed to extract <answer> block: {e}\nModel output:\n{txt}")
        parsed = parse_plan(block)
        return parsed, txt

# -------------------------
# Example run
# -------------------------
def main():
    # define a LunarLander-esque schema
    schema = StateSchema([
        VarSpec("x", -1.5, 1.5),
        VarSpec("y", 0.0, 1.5),
        VarSpec("x_dot", -2.0, 2.0),
        VarSpec("y_dot", -2.0, 2.0),
        VarSpec("theta", -3.14, 3.14, "rad"),
        VarSpec("theta_dot", -4.0, 4.0, "rad/s")
    ])

    planner = TagAnswerPlanner(model_name="microsoft/phi-3-mini-4k-instruct", load_4bit=True)

    instruction = "You are a rocket. Do a spin but make sure to stay in bounds. Then land at 0,0"

    parsed, full_text = planner.generate_plan(instruction, schema, temperature=0.2, top_p=0.9, max_new_tokens=400)
    print("=== FULL MODEL OUTPUT ===\n")
    print(full_text)
    print("\n=== PARSED PLAN (raw) ===")
    print(parsed)

    # ensure trajectory shape
    traj = parsed["trajectory"]
    # naive resample/resize: linear interpolation or tile single step
    if len(traj) > 1:
        new = []
        L = len(traj)
        for t in range(L):
            alpha = t * (L-1) / max(1, L-1)
            lo = int(math.floor(alpha))
            hi = min(L-1, lo+1)
            w = alpha - lo
            row = []
            for v in range(schema.dim):
                lv = traj[lo][v] if v < len(traj[lo]) else 0.0
                hv = traj[hi][v] if v < len(traj[hi]) else lv
                row.append((1-w)*lv + w*hv)
            new.append(row)
        traj = new
    else:
        base = traj[0] if len(traj)>0 else [0.0]*schema.dim
        traj = [ (base + [0.0]*schema.dim)[:schema.dim] for _ in range(L) ]

    # clamp and smooth
    max_delta = [0.08, 0.06, 0.15, 0.15, 0.25, 0.35, 1.0, 1.0]
    traj_feasible = clamp_and_smooth(traj, schema, max_delta=max_delta, smooth_alpha=0.12)

    print("\n=== FINAL TRAJECTORY (clamped & smoothed) ===")
    for t, row in enumerate(traj_feasible):
        print(f"{t:02d}: {[round(x,4) for x in row]}")

    print("\n=== WEIGHTS ===")
    print(parsed.get("weights", None))

if __name__ == "__main__":
    main()
