import gymnasium as gym
import numpy as np
import random
import custom_lunar_lander

# ---------------------------------------------------------------------------
# Helper Functions for Discretization
# ---------------------------------------------------------------------------

def create_bins(low, high, num_bins):
    """
    Create bins for each dimension based on provided lower and upper bounds.
    If the environment's bound is infinite, clip to a pre‑defined value.
    """
    bins = []
    for l, h in zip(low, high):
        # Clip infinite bounds to reasonable finite numbers.
        if not np.isfinite(l):
            l = -10.0
        if not np.isfinite(h):
            h = 10.0
        # Create inner bin edges; this yields `num_bins` discrete intervals.
        bin_edges = np.linspace(l, h, num_bins + 1)[1:-1]
        bins.append(bin_edges)
    return bins

def discretize_state(state, bins):
    """
    Map each dimension of the continuous `state` to a discrete bin index.
    Returns a tuple indexing each dimension.
    """
    discrete_state = tuple(
        int(np.digitize(s, b))
        for s, b in zip(state, bins)
    )
    return discrete_state

def state_to_index(discrete_state, num_bins_vec):
    """
    Map a tuple of discrete state indices into a single integer index.
    This is used for addressing our flat Q-table.
    """
    index = 0
    for i, s in enumerate(discrete_state):
        index = index * num_bins_vec[i] + s
    return index

# ---------------------------------------------------------------------------
# Dyna-Q Functions
# ---------------------------------------------------------------------------

def epsilon_greedy(Q, state_index, n_actions, epsilon):
    """Select an action using the epsilon-greedy method."""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state_index])

def dyna_q(env, bins, num_episodes=200, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=10):
    """
    Dyna-Q algorithm with discretization.

    Parameters:
      env             : The Gymnasium environment.
      bins            : List of numpy arrays with discretization bin edges per dimension.
      num_episodes    : Number of episodes to run training.
      alpha           : Learning rate.
      gamma           : Discount factor.
      epsilon         : Epsilon for epsilon-greedy exploration.
      planning_steps  : Number of simulated (planning) steps per real step.

    Returns:
      Q               : Learned Q-table.
      episode_rewards : List of reward sums per episode.
    """
    # Determine the number of discrete intervals per dimension.
    num_bins_vec = [len(b) + 1 for b in bins]
    total_states = int(np.prod(num_bins_vec))
    n_actions = env.action_space.n

    # Initialize the Q-table and a simple model (dictionary).
    Q = np.zeros((total_states, n_actions))
    model = {}  # Maps (state_index, action) -> (next_state_index, reward, done)
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        discrete_state = discretize_state(obs, bins)
        state_index = state_to_index(discrete_state, num_bins_vec)
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state_index, n_actions, epsilon)
            next_obs, reward, done, truncated, info = env.step(action)
            discrete_next_state = discretize_state(next_obs, bins)
            next_state_index = state_to_index(discrete_next_state, num_bins_vec)
            total_reward += reward

            # Q‑learning update with real experience.
            Q[state_index, action] += alpha * (reward + gamma * np.max(Q[next_state_index]) - Q[state_index, action])
            # Update the model.
            model[(state_index, action)] = (next_state_index, reward, done)
            # Planning step: simulate recent transitions.
            for _ in range(planning_steps):
                if len(model) > 0:
                    (s, a) = random.choice(list(model.keys()))
                    s_next, r, d = model[(s, a)]
                    Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])

            state_index = next_state_index

        episode_rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}")
    return Q, episode_rewards

# ---------------------------------------------------------------------------
# Main: Run Dyna-Q on a Box2D Environment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # For illustration, we use LunarLander-v2 (a Box2D environment). If you have a
    # custom Box2D environment registered (e.g., "CustomBox2D-v0"), replace the ID below.
    env = gym.make("CustomLunarLander-v0", continuous=False)

    # Create bins for discretizing the continuous state.
    # Note: LunarLander-v2 has an 8-dimensional state. We use a small number of bins due
    # to the combinatorial explosion in state space size.
    num_bins_per_dim = 3  # For demonstration; you may experiment with higher resolution.
    bins = create_bins(env.observation_space.low, env.observation_space.high, num_bins_per_dim)

    # Train Dyna-Q.
    Q, rewards = dyna_q(env, bins, num_episodes=300_000, planning_steps=10)
    print("Training completed.\n")

    # Test the learned policy.
    obs, _ = env.reset()
    discrete_state = discretize_state(obs, bins)
    state_index = state_to_index(discrete_state, [len(b)+1 for b in bins])
    total_reward = 0
    done = False

    while not done:
        action = np.argmax(Q[state_index])
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        discrete_state = discretize_state(obs, bins)
        state_index = state_to_index(discrete_state, [len(b)+1 for b in bins])
    print("Test Episode Reward:", total_reward)
