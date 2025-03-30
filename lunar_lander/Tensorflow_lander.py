import tensorflow as tf
import numpy as np

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy

# -------------------------------
# Environment Setup
# -------------------------------

# Load the Gymnasium LunarLander-v2 environment.
# suite_gym.load works well with Gym environments (Gymnasium is its successor).
env_name = "LunarLander-v2"
py_env = suite_gym.load(env_name)
tf_env = tf_py_environment.TFPyEnvironment(py_env)

# Create a separate evaluation environment.
eval_py_env = suite_gym.load(env_name)
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# -------------------------------
# Agent Setup: Q-Network and DQN Agent
# -------------------------------

# Define the Q-Network with one hidden layer (100 units):
fc_layers = (100,)
q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=fc_layers
)

# Create an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Global step counter.
train_step_counter = tf.Variable(0, dtype=tf.int64)

# Initialize the DQN agent. Note that it uses TensorFlow internally.
agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)
agent.initialize()

# -------------------------------
# Replay Buffer and Data Collection
# -------------------------------

# Create a replay buffer with capacity for 100,000 transitions.
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000
)

# A simple function to collect one transition (one step):
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

# --- Warm-up: Collect initial experience with a random policy ---
random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
for _ in range(100):
    collect_step(tf_env, random_policy, replay_buffer)

# Create a dataset from the replay buffer for training.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2
).prefetch(3)
iterator = iter(dataset)

# -------------------------------
# Training Loop
# -------------------------------

num_iterations = 1000
for iteration in range(num_iterations):
    # Collect one transition using the agent's collect policy.
    collect_step(tf_env, agent.collect_policy, replay_buffer)

    # Sample a batch of experience and train the agent.
    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss

    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Loss = {train_loss:.4f}")

# -------------------------------
# Evaluation: Run a Single Episode
# -------------------------------

num_episodes = 1
returns = []

for _ in range(num_episodes):
    time_step = eval_tf_env.reset()
    episode_return = 0.0
    # Run until the episode ends
    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        time_step = eval_tf_env.step(action_step.action)
        # Since rewards are batched, take the first element from the reward array
        episode_return += time_step.reward.numpy()[0]
    returns.append(episode_return)

print("Final evaluation return:", np.mean(returns))
