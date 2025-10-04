from unsloth import FastLanguageModel
from json import load
from interpretors.Interpretor import Interpretor
import interpretors.parsing as parsing

class GPT_OSS_Model(Interpretor):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.messages = []

    # def read_config(self):
    #     with open('config/config.json', 'r') as file:
    #         config = load(file)
    #     return config

    def load(self):
        max_seq_length = 2048
        dtype = None

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/gpt-oss-20b",
            dtype = dtype, # None for auto detection
            max_seq_length = max_seq_length, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            
            # full_finetuning = False, # [NEW!] We have full finetuning now!
            # token = "hf_...", # use one if using gated models
        )

    def give_command(self, user_command, new=True):
        assert(self.model)
        assert(self.tokenizer)

        if (new):
            print("new chat")
            self.messages = []
            system_prompt = f'''{self.config.get("context")}.
Give a trajectory of states and weights for the following command.
The trajectory must have sufficient although minimal target states (possibly 1) to capture the complete expected behaviour. The states do not need to be over a complete time sequence, but just represent targets.
Each state variable in each state must have a weight value between 0 an 1 depending on how important it is to capture the desired user behaviour. VERY IMPORTANT: The weights can be different across target states if the individual state variable priorities change.
The states are {self.config.get("states")} with max and min values of {self.config.get("bounds").get("max")} and {self.config.get("bounds").get("min")}. Do not exceed these bounds.
Reasoning should be given for each state and weighting and be presented in the <reasoning> </reasoning> tags.
Trajectory should be given in the <trajectory></trajectory> tags and be presented as a 2d array of states.
Weights should be given in the <weights></weights> tags and be presented as a 2d array of weights. There must be as many weights as states in the trajectory.
The order should be <reasoning>, <trajectory>, <weights>
The <weights> and <trajectory> should only be a 2d array of floats, no comments or expressions.
The current state is {self.config.get("current_state")}. You may need to move to an appropriate starting state before beginning the command.
VERY IMPORTANT: If the instruction is too vague or anytime you feel you need to make an assumption, simply stop and output only <clarification> clarification_message </clarification> for the user to clarify certain aspects.
DO NOT GUESS ANYTHING
I will present the instructions as line by line commands.
            '''
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
