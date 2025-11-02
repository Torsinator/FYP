import numpy as np
import copy
from pysindy import PolynomialLibrary, FourierLibrary, GeneralizedLibrary
import pysindy as ps
import gymnasium as gym
import custom_lunar_lander_no_target
import custom_environment_wrapper
from stable_baselines3 import SAC

# Import sindy_rl components
from sindy_rl.env import rollout_env
from sindy_rl.policy import RandomPolicy, BasePolicy
from sindy_rl.dynamics import EnsembleSINDyDynamicsModel

# Import custom components
from Sindy_Env_Wrapper import LunarLanderSindy
from mpc import MPCPolicy
from uncertainty_explorer import find_high_uncertainty_states
from coupled_fourier_library import CoupledFourierLibrary

class RLPolicy(BasePolicy):
    '''
    A random policy
    '''
    def __init__(self, model, seed=0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        self.model = model
        
    def compute_action(self, obs):
        a,_ = model.predict(obs)
        return a
    
    def set_magnitude_(self, mag):
        self.magnitude = mag

# Map Gym state to MPC state format
def gym_state_to_mpc(self, state):
    x, y, theta, vx, vy, omega, *_ = state
    # Multipliers from Lunar Lander
    return np.array([x * 10, y * 6.666, vx * 5, vy * 7.5, theta, omega * 2.5])

def three_state_reward(state, target_state, weights):
    distance = np.sqrt(np.sum(weights * (state[:3] - target_state[[0,1,4]]) ** 2))
    reward = -distance
    if distance < 0.01:
        reward += 10
    return reward

def three_state_obs(obs, target_state, weights):
    pos_obs = obs[[0,1,4]]
    return np.concatenate((pos_obs, obs[[2,3,5]]), dtype=np.float32)
    # return np.concatenate((pos_obs, obs[[2,3,5]]), dtype=np.float32)

def target_state_fn(obs_space):
    low = np.array([-1, -0.2, -2*np.pi])
    high = np.array([1, 1.5, 2*np.pi])
    return np.array([0, 0, 0])
    # return np.random.uniform(low, high)
    while True:
        mask = np.random.random(3) < 0.5
        result = mask * np.random.uniform(low, high)
        if np.any(result != 0):
            return result

def weights_generation_fn():
    while True:
        mask = np.random.random(3) < 0.5
        result = mask * np.random.random(3)
        if np.any(result != 0):
            print(result / np.max(result))
            return result / np.max(result)

# --- Parallel environment ---
def make_env(seed=None):
    def _init():
        env = gym.make("CustomLunarLander-v0", continuous=True)
        env = custom_environment_wrapper.CustomEnvironmentWrapper(env, three_state_obs, three_state_reward,
                                        target_state_fn, weights_generation_fn)
        # if seed is not None:
        #     env.seed(seed)
        return env
    return _init()

env = LunarLanderSindy.make_env()
# model = SAC.load("models/SAC_Unweighted_9_States_Sparse", env=env)

random_policy = RandomPolicy(env.action_space)

# Build combined feature library (polynomial + Fourier terms)
combined_library = GeneralizedLibrary([
    PolynomialLibrary(degree=1),
    CoupledFourierLibrary(2)
    # FourierLibrary(n_frequencies=1)
])

# Define dynamics configuration
dyna_config = {
    'dt': 0.02,
    'discrete': True,  # Discrete-time formulation
    'optimizer': {
      'base_optimizer': {
        'name': 'STLSQ',
        'kwargs': {
          'alpha': 5.0e-5,
          'threshold': 0.005,
            },
      },
        # Ensemble Optimization config
      'ensemble': {
        'bagging': True,
        'library_ensemble': True,
        'n_models': 20,
      },
    },
        # Dictionary/Libary Config
    'feature_library': combined_library
}

# Initialize the ensemble SINDy model
dyn_model = EnsembleSINDyDynamicsModel(dyna_config)

# Create the environment
# random_policy = RLPolicy(model)

# Collect an initial dataset
traj_obs, traj_acts, traj_rews = rollout_env(env, random_policy, n_steps=10000, n_steps_reset=100)

train_obs = traj_obs[1:]
test_obs = traj_obs[0]

train_acts = traj_acts[1:]
test_acts = traj_acts[0]
dyn_model.fit(train_obs, train_acts)

dyn_model.set_median_coef_()
dyn_model.print()

median_obs = [test_obs[0]]

for u in test_acts:
    x = median_obs[-1]
    x_new = dyn_model.predict(x, u)
    median_obs.append(x_new)

median_obs = np.array(median_obs)

# Build CasADi-compatible feature list for MPC
features = dyn_model.model.get_feature_names()
casadi_features = []
for f in features:
    expr = f.replace(" ", "*").replace("^", "**")
    expr = expr.replace("sin", "ca.sin").replace("cos", "ca.cos")
    casadi_features.append(expr)

coeffs = dyn_model.model.coefficients()
library = casadi_features

rhs_exprs = []
for term in coeffs:
    expr_str = ""
    for i in range(len(term)):
        if abs(float(term[i])) > 0.005:
            if len(expr_str) > 0:
                expr_str += " + "
            expr_str += f"{term[i]}*{library[i]}"
    rhs_exprs.append(expr_str)

file_name = "odes.txt"
with open(file_name, "w") as file:
    for item in rhs_exprs:
        file.write(item + "\n")

# again for the mean observation
dyn_model.set_mean_coef_()
mean_obs = [test_obs[0]]

for u in test_acts:
    x = mean_obs[-1]
    x_new = dyn_model.predict(x, u)
    mean_obs.append(x_new)

mean_obs = np.array(mean_obs)

from matplotlib import pyplot as plt

fig, axes  = plt.subplots(3,2, figsize=(12,20))

plt_labels = [r'$x$', r'$y$', r'$vx$', r'$vy$', r'$\theta$', r'$\omega$']

for i, ax in enumerate(axes.flatten()):

    ax.plot(test_obs[:,i], 'k--', label = 'test_data')
    ax.plot(median_obs[:, i], label = 'median preds')
    ax.plot(mean_obs[:, i], label = 'mean preds')
    ax.set_title(plt_labels[i])
    ax.legend()

plt.show()
dyn_model.save("pysindy_save.txt")

target_states, uncertainty_scores = find_high_uncertainty_states(
        dyn_model,
        traj_obs,
        traj_acts,
        n_states=5,
        horizon=1,
        n_samples=100,
        uncertainty_threshold=0.7
    )

print(target_states, uncertainty_scores)
