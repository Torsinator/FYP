from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import ast

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

def generate_state(user_command):
    prompt = f"""
        I have a state vector with six components: [x, y, vx, vy, angle, angular_velocity]. For a rocket

        The meanings are:
        - x, y: target coordinates.
        - vx, vy: velocities (default 0 if not given).
        - angle: orientation of the rocket.
        - angular_velocity: rotational speed. How the angle changes over time (default 0 if not given).

        Examples:
        - Command: "land at coordinates (3.5, -4.2)" should produce: [3.5, -4.2, 0, 0, 0, 0].
        - Command: "spin at 2 rad/s" should produce: [0, 0, 0, 0, 0, 2].

        Now, I want you convert the following human command:
        '{user_command}'
        into the corresponding state vector with 6 elements as described above:
        """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = generated_text[len(prompt):]  # Remove the prompt part
    print(output_text)
    # Example string representing a state vector
    index = output_text.find(']')
    state_string = output_text[:index + 1]

    # Use ast.literal_eval to safely evaluate the string and convert to a list
    state_list = ast.literal_eval(state_string)

    # Convert the list to a NumPy array
    state_vector = np.array(state_list, dtype=np.float32)
    print(state_vector)
    return state_vector