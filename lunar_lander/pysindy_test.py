from sindy_rl.dynamics import EnsembleSINDyDynamicsModel
from Sindy_Env_Wrapper import LunarLanderSindy
from pysindy import PolynomialLibrary
from pysindy import FourierLibrary
import pysindy as ps
import mpc
import numpy as np
from mpc import MPCPolicy
import copy
from uncertainty_explorer import find_high_uncertainty_states

combined_library = ps.GeneralizedLibrary([
    ps.PolynomialLibrary(degree=2),
    ps.FourierLibrary(n_frequencies=2)
])

dyna_config = {
    'dt': 0.02,
    'discrete': False,
    
    # Optimizer config 
    'optimizer': {
      'base_optimizer': {
        'name': 'STLSQ',
        'kwargs': {
        #   'alpha': 5.0e-5,
          'threshold': 0.1,
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

dyn_model = EnsembleSINDyDynamicsModel(dyna_config)

from sindy_rl.env import rollout_env
from sindy_rl.policy import RandomPolicy

env = LunarLanderSindy.make_env()


random_policy = RandomPolicy(env.action_space)
traj_obs, traj_acts, traj_rews = rollout_env(env, random_policy, n_steps = 20000, n_steps_reset=1000)

for i in range(3):

  train_obs = traj_obs[:-1]
  test_obs = traj_obs[-1]

  train_acts = traj_acts[:-1]
  test_acts = traj_acts[-1]
  dyn_model.fit(train_obs, train_acts)

  print(dyn_model.set_median_coef_())
  features = dyn_model.model.get_feature_names()
  casadi_features = []
  for symbol in features:
      new_symbol = ""
      for letter in symbol:
          if letter.isspace():
              new_symbol += "*"
          elif letter == "^":
              new_symbol += "**"
          else:
              new_symbol += letter
      new_symbol = new_symbol.replace("sin", "ca.sin")
      new_symbol = new_symbol.replace("cos", "ca.cos")
      casadi_features.append(new_symbol)

  print(casadi_features)
      
  dyn_model.print()
  # with open("coefs.txt", "w+") as file:
  #   file.write(str(dyn_model.get_coef_list()))
  # print(features)
  print(dyn_model.model.coefficients())
  # exit()


  target_states, uncertainty_scores = find_high_uncertainty_states(
    dyn_model, traj_obs, traj_acts, 
    n_states=5,          # How many states to explore
    horizon=10,          # Multi-step rollout uncertainty
    n_samples=100,
    uncertainty_threshold=0.7  # Top 30% most uncertain
    )

  # mpc_model = mpc.create_lander_model(casadi_features, dyn_model.model.coefficients())
  # mpc.run_episode(mpc_model, states)

  mpc_policy = MPCPolicy(casadi_features, dyn_model.model.coefficients())
  new_obs, new_acts, new_rews = mpc_policy.rollout_env(env, target_states)
  # # assert(len(new_obs[0]) == len(new_acts[0]))
  # # print("traj_obs", np.shape(traj_obs))
  # # print("new_obs", np.shape(new_obs))
  assert(len(traj_obs) == len(traj_acts))
  (traj_obs.append(i) for i in copy.deepcopy(new_obs))
  (traj_acts.append(i) for i in copy.deepcopy(new_acts))
#   traj_obs = np.vstack((traj_obs, new_obs))
#   traj_acts = np.vstack((traj_acts, new_acts))
  # print(len(new_obs[0]))
  # traj_obs, traj_acts, traj_rews = rollout_env(env, random_policy, n_steps = 1000, n_steps_reset=1000)



import numpy as np
median_obs = [test_obs[0]]

for u in test_acts:
    x = median_obs[-1]
    x_new = dyn_model.predict(x, u)
    median_obs.append(x_new)

median_obs = np.array(median_obs)



# again for the mean observation
dyn_model.set_mean_coef_()
mean_obs = [test_obs[0]]

for u in test_acts:
    x = mean_obs[-1]
    x_new = dyn_model.predict(x, u)
    mean_obs.append(x_new)

mean_obs = np.array(mean_obs)



from matplotlib import pyplot as plt

fig, axes  = plt.subplots(1,6, figsize=(25,6))

plt_labels = [r'$x$', r'$y$', r'$vx$', r'$vy$', r'$\theta$', r'$\omega$']

for i, ax in enumerate(axes.flatten()):

    ax.plot(test_obs[:,i], 'k--', label = 'test_data')
    ax.plot(median_obs[:, i], label = 'median preds')
    ax.plot(mean_obs[:, i], label = 'mean preds')
    ax.set_title(plt_labels[i])
    ax.legend()



from tqdm import tqdm

all_preds = []

for idx in tqdm(range(20)):
    dyn_model.set_idx_coef_(idx)
    obs_list = [test_obs[0]]

    for u in test_acts:
        x = obs_list[-1]
        try:
            x_new = dyn_model.predict(x, u)
            obs_list.append(x_new)
        except ValueError:
            print('!! Integration blew up !!') 
            # backtrak just for plotting purposes.
            obs_list = obs_list[:-10]
            break
        
    obs_list = np.array(obs_list)
    all_preds.append(obs_list)



from matplotlib import pyplot as plt

fig, axes  = plt.subplots(1,6, figsize=(25,6))

for i, ax in enumerate(axes.flatten()):

    ax.plot(test_obs[:,i], 'k--', label = 'test_data')
    ax.plot(mean_obs[:, i], label = 'mean preds')
    
    for pred_obs in all_preds:
        ax.plot(pred_obs[:, i], c='r', alpha = 0.1)
    ax.legend()
    ax.set_title(plt_labels[i])

plt.show()

dyn_model.save("pysindy_save.txt")