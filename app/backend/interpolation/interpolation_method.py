from scipy.interpolate import interp1d
import numpy as np

def interpolate_(values, episode_length, hz):
    """
    Interpolate target values so that we have one target per timestep.
    values: array [N, state_dim]
    """
    num_waypoints = len(values)
    state_dim = values.shape[1]

    # Original timepoints (spread across episode)
    t_waypoints = np.linspace(0, episode_length, num_waypoints)

    # New dense timeline at desired resolution
    t_dense = np.linspace(0, episode_length, int(episode_length * hz))

    # Interpolate each dimension separately
    values_interp = np.zeros((len(t_dense), state_dim))
    for d in range(state_dim):
        f = interp1d(t_waypoints, values[:, d], kind="linear")
        values_interp[:, d] = f(t_dense)

    return values_interp

def interpolate():
    functions = [inter]