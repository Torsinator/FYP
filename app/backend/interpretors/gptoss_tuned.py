from unsloth import FastLanguageModel
from json import load
from interpretors.Interpretor import Interpretor
import interpretors.parsing as parsing

class GPT_OSS_Model_Tuned(Interpretor):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.messages = []

    def load(self):
        max_seq_length = 2048
        dtype = None

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "./llm/planning_model",
            dtype = dtype, # None for auto detection
            max_seq_length = max_seq_length, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            
            # full_finetuning = False, # [NEW!] We have full finetuning now!
            # token = "hf_...", # use one if using gated models
        )

        FastLanguageModel.for_inference(self.model)

    def give_command(self, user_command, new=True):
        assert(self.model)
        assert(self.tokenizer)

        if (new):
            print("new chat")
            self.messages = []
            system_prompt = f"""
{self.config.get("context")}

TASK
Generate a minimal, sufficient **trajectory** of target states and a matching 2D array of per-variable **weights** for the given command.

DEFINITIONS
States: {self.config.get("states")}
Bounds: min = {self.config.get("bounds").get("min")}, max = {self.config.get("bounds").get("max")}
Current state: {self.config.get("current_state")}

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
     - The current state does not influence future weights
     - No extra text or formatting.

VALIDATION STEPS (you must perform these checks before returning)
1. Shape: number of columns in each `<trajectory>` row == number of state variables in `States`. Number of rows in `<weights>` == number of rows in `<trajectory>`. Each corresponding row length must match.
2. Bounds: every trajectory value must satisfy `min <= value <= max` for the corresponding state variable.
3. Weights: every weight must satisfy `0.0 <= weight <= 1.0`.
4. Formatting: `<trajectory>` and `<weights>` must be valid JSON arrays containing only numeric literals (no comments, no text).
If any check fails, **do not** output reasoning or arrays — output **only** a `<clarification>` tag listing the failing checks (short, comma-separated).

ADDITIONAL RULES
- Do not include any other tags or text outside the tags described above.
- Do not guess: if you must assume something to proceed, stop and request clarification using `<clarification></clarification>`.
- You may include an initial state in `<trajectory>` to represent moving from `Current state` to the first target if appropriate.

EXAMPLE (format only — replace with real numbers that respect bounds and shapes)
<reasoning>Concise reason for states and weights.</reasoning>
<trajectory>[[0.0, 1.0, 0.5], [0.2, 0.9, 0.1]]</trajectory>
<weights>[[1.0, 0.8, 0.2], [0.9, 0.7, 0.1]]</weights>

Now produce the output for the command that follows.
"""

            self.messages.append({"role": "system", "content": f"{system_prompt}"})

        self.messages.append({"role": "user", "content": f"{user_command}"})
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt = True,
            return_tensors = "pt",
            return_dict = True,
            reasoning_effort = "low",
        ).to(self.model.device)
        from transformers import TextStreamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        outputs = self.model.generate(**inputs, max_new_tokens = 4096, do_sample = False, streamer = streamer,)
        generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(generated_text)
        self.messages.append({"role": "assistant", "content": f"{generated_text}"})
        return parsing.parse_output(generated_text, self.config)
