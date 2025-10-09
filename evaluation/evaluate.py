import pandas as pd
import numpy as np

import agents.PPO as PPO

targets = pd.read_csv("datasets/targets.csv")

agents = [PPO.PPO_Unweighted_9_States]

agent, env = PPO.PPO_Unweighted_9_States.load_model()

results = []

def calculate_error(target, current):
    return np.linalg.norm(target - current, axis=-1), target - current

for i, target_state in targets.iterrows():
    target_state = np.array([target_state.x, target_state.y, target_state.theta], dtype=np.float32)
    obs, info = env.reset()
    agent.target = target_state
    done = False
    error_vals = [np.inf, np.inf, np.inf]
    lowest_error = np.inf
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        print(action)
        obs, reward, completed, terminated, info = env.step(action)
        done = completed or terminated
        error, vals = calculate_error(target_state, obs[[0,1,2]])
        if error < lowest_error:
            lowest_error = error
            error_vals = vals
    addition = np.concatenate([[lowest_error], error_vals])
    results.append(addition)

column_names = ['error', 'dx', 'dy', 'dtheta']
df_custom = pd.DataFrame(results, columns=column_names)

    

