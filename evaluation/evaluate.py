import pandas as pd
import numpy as np

import agents.PPO as PPO
import agents.TD3 as TD3
import agents.SAC as SAC

targets = pd.read_csv("datasets/targets.csv")

models = [PPO.PPO_Unweighted_9_States]

model = SAC.SAC_Unweighted_9_States

agent, env = model.load_model()

results = []

def calculate_error(target, current):
    return np.linalg.norm(target - current, axis=-1), target - current

for i, target_state in targets.iterrows():
    target_state = np.array([target_state.x, target_state.y, target_state.theta], dtype=np.float32)
    env.target = target_state
    obs, info = env.reset()
    done = False
    error_vals = [np.inf, np.inf, np.inf]
    lowest_error = np.inf
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, completed, terminated, info = env.step(action)
        done = completed or terminated
        if isinstance(obs, dict):
            obs_vec = obs.get("observation", obs)
        else:
            obs_vec = obs
        error, vals = calculate_error(target_state, obs_vec[[0,1,2]])
        if error < lowest_error:
            lowest_error = error
            error_vals = vals
    addition = np.concatenate([target_state, [lowest_error], error_vals])
    print(lowest_error)
    results.append(addition)

column_names = ['tx', 'ty', 'ttheta', 'error', 'dx', 'dy', 'dtheta']
df_custom = pd.DataFrame(results, columns=column_names)

import matplotlib.pyplot as plt
import seaborn as sns

# Overall error distribution
plt.figure(figsize=(6,4))
plt.plot(df_custom["error"], "o")
plt.title("Distribution of Final Tracking Error")
# plt.show()

mean = np.mean(df_custom["error"])
var = np.var(df_custom["error"])
df_custom.to_csv(f"results/{model.__name__}.csv", index=False)
print(mean, var)



