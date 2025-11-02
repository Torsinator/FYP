from sindy_rl.dynamics import EnsembleSINDyDynamicsModel

dyna_config = {
    'callbacks': 'project_cartpole',
    'dt': 1,
    'discrete': True,
    
    # Optimizer config 
    'optimizer': {
      'base_optimizer': {
        'name': 'STLSQ',
        'kwargs': {
          'alpha': 5.0e-5,
          'threshold': 7.0e-3,
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
    'feature_library': {
      'name': 'affine', # use affine functions
      'kwargs': {
        'poly_deg': 2,
        'n_state': 5 ,
        'n_control': 1,
        'poly_int': True,
        'tensor': True,
      }
    }
}

dyn_model = EnsembleSINDyDynamicsModel(dyna_config)

from sindy_rl.env import rollout_env
from sindy_rl.registry import DMCEnvWrapper
from sindy_rl.policy import RandomPolicy

env_config = {'domain_name': "cartpole",
                'task_name': "swingup",
                'frame_skip': 1,
                'from_pixels': False}
cart_env = DMCEnvWrapper(env_config)


random_policy = RandomPolicy(cart_env.action_space)
traj_obs, traj_acts, traj_rews = rollout_env(cart_env, random_policy, n_steps = 8000, n_steps_reset=1000)



train_obs = traj_obs[:-1]
test_obs = traj_obs[-1]

train_acts = traj_acts[:-1]
test_acts = traj_acts[-1]
dyn_model.fit(train_obs, train_acts)

dyn_model.set_median_coef_()
dyn_model.print()



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

fig, axes  = plt.subplots(1,5, figsize=(25,5))

plt_labels = [r'$x$', r'$\cos(\theta)$', r'$\sin(\theta)$', r'$dx/dt$', r'$d \theta / dt$']

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

fig, axes  = plt.subplots(1,5, figsize=(25,5))

for i, ax in enumerate(axes.flatten()):

    ax.plot(test_obs[:,i], 'k--', label = 'test_data')
    ax.plot(median_obs[:, i], label = 'median preds')
    
    for pred_obs in all_preds:
        ax.plot(pred_obs[:, i], c='r', alpha = 0.1)
    ax.legend()
    ax.set_title(plt_labels[i])

plt.show()