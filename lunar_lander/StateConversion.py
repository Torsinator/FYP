# Ramer Douglas Pecker Algorithm (Multivariate)

import numpy as np
import custom_lunar_lander
import gymnasium as gym

np.set_printoptions(precision=2, suppress=True)

# Generated using microsoft copilot
# Sine wave pattern 1Hz
states = np.array(
  [
  [-1.00000, 0.00000, 0.20000, 0.00000, 0.00000, 10.00000],
  [-0.95000, 1.00000, 0.20000, 6.28319, 1.53980, 0.00000],
  [-0.90000, 2.00000, 0.20000, 0.00000, 0.00000, -10.00000],
  [-0.85000, 1.00000, 0.20000, -6.28319, -1.53980, 0.00000],
  [-0.80000, 0.00000, 0.20000, 0.00000, 0.00000, 10.00000],
  [-0.75000, 1.00000, 0.20000, 6.28319, 1.53980, 0.00000],
  [-0.70000, 2.00000, 0.20000, 0.00000, 0.00000, -10.00000],
  [-0.65000, 1.00000, 0.20000, -6.28319, -1.53980, 0.00000],
  [-0.60000, 0.00000, 0.20000, 0.00000, 0.00000, 10.00000],
  [-0.55000, 1.00000, 0.20000, 6.28319, 1.53980, 0.00000],
  [-0.50000, 2.00000, 0.20000, 0.00000, 0.00000, -10.00000],
  [-0.45000, 1.00000, 0.20000, -6.28319, -1.53980, 0.00000],
  [-0.40000, 0.00000, 0.20000, 0.00000, 0.00000, 10.00000],
  [-0.35000, 1.00000, 0.20000, 6.28319, 1.53980, 0.00000],
  [-0.30000, 2.00000, 0.20000, 0.00000, 0.00000, -10.00000],
  [-0.25000, 1.00000, 0.20000, -6.28319, -1.53980, 0.00000],
  [-0.20000, 0.00000, 0.20000, 0.00000, 0.00000, 10.00000],
  [-0.15000, 1.00000, 0.20000, 6.28319, 1.53980, 0.00000],
  [-0.10000, 2.00000, 0.20000, 0.00000, 0.00000, -10.00000],
  [-0.05000, 1.00000, 0.20000, -6.28319, -1.53980, 0.00000],
  [0.00000, 0.00000, 0.20000, 0.00000, 0.00000, 10.00000],
  [0.05000, 1.00000, 0.20000, 6.28319, 1.53980, 0.00000],
  [0.10000, 2.00000, 0.20000, 0.00000, 0.00000, -10.00000],
  [0.15000, 1.00000, 0.20000, -6.28319, -1.53980, 0.00000],
  [0.20000, 0.00000, 0.20000, 0.00000, 0.00000, 10.00000],
  [0.25000, 1.00000, 0.20000, 6.28319, 1.53980, 0.00000],
  [0.30000, 2.00000, 0.20000, 0.00000, 0.00000, -10.00000],
  [0.35000, 1.00000, 0.20000, -6.28319, -1.53980, 0.00000],
  [0.40000, 0.00000, 0.20000, 0.00000, 0.00000, 10.00000],
  [0.45000, 1.00000, 0.20000, 6.28319, 1.53980, 0.00000],
  [0.50000, 2.00000, 0.20000, 0.00000, 0.00000, -10.00000],
  [0.55000, 1.00000, 0.20000, -6.28319, -1.53980, 0.00000],
  [0.60000, 0.00000, 0.20000, 0.00000, 0.00000, 10.00000],
  [0.65000, 1.00000, 0.20000, 6.28319, 1.53980, 0.00000],
  [0.70000, 2.00000, 0.20000, 0.00000, 0.00000, -10.00000]]
)

env = gym.make("CustomLunarLander-v0", continuous=False)
LOW = env.observation_space.low
HIGH = env.observation_space.high

def normalise(states):
    for state in states:
        for i in range(len(state)):
            state[i] /= ((HIGH[i] - LOW[i]) / 2)


def rdp(states, eps):
    START = states[0]
    END = states[-1]
    LINE_VECTOR = END - START
    d_max = 0
    i_max = 0
    for i in range(1, len(states) - 1):
        state = states[i]
        t = np.dot(state - START, LINE_VECTOR) / (np.linalg.norm(END - START))**2
        t = max(0, min(t, 1))   # make sure t is in range [0, 1]
        proj = START + t * LINE_VECTOR
        d = np.linalg.norm(state - proj)    # distance from straight line between A and B
        if (d > d_max):
            d_max, i_max = d, i

    if d_max > eps:
        # Both left and right have the i_max point
        left = rdp(states[:i_max + 1], eps)
        right = rdp(states[i_max:], eps)
        return left + right[1:]    # make sure the mid point is not duplicated
    else:
        # base case, the max distance is less than the threshold
        return [START, END]

eps = 0.7
normalise(states)
new_states = rdp(states, eps)
print(new_states)
print(f"Reduced length from {len(states)}, to {len(new_states)}")
