from agents.SAC import *
from agents.PPO import *
from stable_baselines3.common.callbacks import CheckpointCallback

import numpy as np

def generate_value_pretraining_data(env, n_samples=50000):
    obs_dim = env.observation_space.shape[0]
    data = np.random.uniform(low=env.observation_space.low,
                             high=env.observation_space.high,
                             size=(n_samples, obs_dim))

    # Example: distance from goal (assuming env.target is set)
    targets = []
    for obs in data:
        dist = np.linalg.norm(obs[:3] - env.target[:3])
        value = -dist  # closer = higher value
        targets.append(value)
    return data, np.array(targets, dtype=np.float32)

train_models = [
    # SAC_Unweighted_9_States_Sparse, 
    # SAC_Unweighted_9_States_Dense,
    # SAC_Weighted_9_States_Sparse, 
    SAC_Weighted_9_States_Dense,
    # SAC_Weighted_6_States_Sparse, 
    # SAC_Weighted_6_States_Dense,
    # PPO_Unweighted_9_States_Sparse, 
    # PPO_Unweighted_9_States_Dense,
    # PPO_Unweighted_6_States_Sparse, 
    # PPO_Unweighted_6_States_Dense,
    # PPO_Weighted_9_States_Sparse, 
    # PPO_Weighted_9_States_Dense,
    # PPO_Weighted_6_States_Sparse, 
    # PPO_Weighted_6_States_Dense,
    # SAC_Unweighted_6_States_Sparse, 
    # SAC_Unweighted_6_States_Dense,
]


num_steps = 700_000

checkpoint = 1_000_000

for cls in train_models:
    # Instantiate the checkpoint callback
    checkpoint_callback = CheckpointCallback(
    save_freq=checkpoint,
    save_path='./checkpoints/',
    name_prefix=cls.__name__
    )
    model, env = cls.create_model()
    model.learn(num_steps)
    model.save(f"models/{cls.__name__}")