import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def find_high_uncertainty_states(dyn_model, traj_obs, traj_acts, 
                                  n_states=5, 
                                  n_samples=500,  # Reduced default
                                  horizon=1,  # Single step by default
                                  uncertainty_threshold=0.7):
    """
    Identify states with highest ensemble uncertainty for targeted exploration.
    
    Parameters:
    -----------
    dyn_model : EnsembleSINDyDynamicsModel
        Trained ensemble dynamics model
    traj_obs : list of arrays
        List of trajectory observations
    traj_acts : list of arrays  
        List of trajectory actions
    n_states : int
        Number of high-uncertainty states to return
    n_samples : int
        Number of state-action pairs to sample for uncertainty estimation
    horizon : int
        Prediction horizon for rollout uncertainty (1=fast, >1=more accurate but slower)
    uncertainty_threshold : float
        Percentile threshold (0-1) for filtering high uncertainty regions
        
    Returns:
    --------
    target_states : np.array
        Array of shape (n_states, state_dim) with high-uncertainty initial states
    uncertainty_scores : np.array
        Uncertainty score for each target state
    """
    
    # Flatten all observations and actions
    all_obs = np.vstack([obs for obs in traj_obs])
    all_acts = np.vstack([acts for acts in traj_acts])
    
    # Sample random state-action pairs from collected data
    n_data = min(n_samples, len(all_obs) - 1)
    indices = np.random.choice(len(all_obs) - 1, n_data, replace=False)
    sample_obs = all_obs[indices]
    sample_acts = all_acts[indices]
    
    print(f"Computing uncertainty over {n_data} samples with {20} ensemble models...")
    
    # Vectorized ensemble predictions
    ensemble_predictions = np.zeros((20, n_data, sample_obs.shape[1]))
    
    from tqdm import tqdm
    for idx in tqdm(range(20), desc="Ensemble models"):
        dyn_model.set_idx_coef_(idx)
        
        # Batch predict if possible, otherwise loop
        for i in range(n_data):
            try:
                ensemble_predictions[idx, i] = dyn_model.predict(sample_obs[i], sample_acts[i])
            except (ValueError, RuntimeError):
                ensemble_predictions[idx, i] = np.nan
    
    # Compute uncertainty as std across ensemble for each state
    state_uncertainties = np.nanstd(ensemble_predictions, axis=0)  # (n_samples, state_dim)
    
    # Aggregate uncertainty across state dimensions
    total_uncertainty = np.nanmean(state_uncertainties, axis=1)  # (n_samples,)
    
    # Multi-step rollout uncertainty (optional, more expensive)
    if horizon > 1:
        print(f"Computing {horizon}-step rollout uncertainty (this may take a while)...")
        rollout_uncertainties = compute_rollout_uncertainty(
            dyn_model, sample_obs, sample_acts, horizon, n_samples=min(100, n_data)
        )
        # Combine immediate and rollout uncertainty
        total_uncertainty[:len(rollout_uncertainties)] = (
            0.5 * total_uncertainty[:len(rollout_uncertainties)] + 
            0.5 * rollout_uncertainties
        )
    
    # Filter to high-uncertainty regions (top percentile)
    threshold_value = np.nanpercentile(total_uncertainty, uncertainty_threshold * 100)
    high_uncertainty_mask = total_uncertainty >= threshold_value
    
    high_unc_states = sample_obs[high_uncertainty_mask]
    high_unc_scores = total_uncertainty[high_uncertainty_mask]
    
    print(f"Found {len(high_unc_states)} states above uncertainty threshold ({threshold_value:.4f})")
    
    # Cluster high-uncertainty states to get diverse exploration targets
    if len(high_unc_states) > n_states:
        # Use K-means to find representative states
        kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        kmeans.fit(high_unc_states)
        
        # Get cluster centers as target states
        target_states = kmeans.cluster_centers_
        
        # Assign uncertainty scores (max uncertainty in each cluster)
        labels = kmeans.labels_
        uncertainty_scores = np.array([
            np.max(high_unc_scores[labels == i]) for i in range(n_states)
        ])
    else:
        # If we have fewer high-uncertainty states than requested, return all
        target_states = high_unc_states
        uncertainty_scores = high_unc_scores
        
        # Pad with random states if needed
        if len(target_states) < n_states:
            n_random = n_states - len(target_states)
            random_states = all_obs[np.random.choice(len(all_obs), n_random)]
            target_states = np.vstack([target_states, random_states])
            uncertainty_scores = np.concatenate([
                uncertainty_scores, 
                np.zeros(n_random)
            ])
    
    # Sort by uncertainty (highest first)
    sorted_indices = np.argsort(uncertainty_scores)[::-1]
    target_states = target_states[sorted_indices]
    uncertainty_scores = uncertainty_scores[sorted_indices]
    
    print(f"\nTop {n_states} uncertainty scores:")
    for i, score in enumerate(uncertainty_scores):
        print(f"  State {i+1}: uncertainty = {score:.4f}")
    
    return target_states, uncertainty_scores


def compute_rollout_uncertainty(dyn_model, initial_states, actions, horizon, n_samples=100):
    """
    Compute uncertainty over multi-step rollouts (expensive - use sparingly).
    
    Parameters:
    -----------
    dyn_model : EnsembleSINDyDynamicsModel
        Trained ensemble dynamics model
    initial_states : np.array
        Initial states to rollout from (n_samples, state_dim)
    actions : np.array
        Actions to apply (n_samples, action_dim)
    horizon : int
        Number of steps to rollout
    n_samples : int
        Number of samples to use (subset for speed)
        
    Returns:
    --------
    rollout_uncertainties : np.array
        Average uncertainty over the rollout horizon (n_samples,)
    """
    # Subsample for speed
    n = min(n_samples, len(initial_states))
    indices = np.random.choice(len(initial_states), n, replace=False)
    states_subset = initial_states[indices]
    actions_subset = actions[indices]
    
    all_trajectories = []
    
    from tqdm import tqdm
    for idx in tqdm(range(20), desc="Rollout uncertainty"):
        dyn_model.set_idx_coef_(idx)
        
        # Rollout this model
        states = states_subset.copy()
        model_trajectory = [states]
        
        for h in range(min(horizon, len(actions_subset))):
            next_states = np.zeros_like(states)
            for i, (state, action) in enumerate(zip(states, actions_subset)):
                try:
                    next_states[i] = dyn_model.predict(state, action)
                except (ValueError, RuntimeError):
                    next_states[i] = np.nan
            
            states = next_states
            model_trajectory.append(states)
        
        all_trajectories.append(np.array(model_trajectory))
    
    # Compute uncertainty across ensemble trajectories
    all_trajectories = np.array(all_trajectories)  # (n_models, horizon+1, n_samples, state_dim)
    
    cumulative_uncertainty = np.zeros(n)
    # Average uncertainty over time horizon
    for h in range(1, min(horizon + 1, all_trajectories.shape[1])):
        step_uncertainty = np.nanstd(all_trajectories[:, h, :, :], axis=0)  # (n_samples, state_dim)
        cumulative_uncertainty += np.nanmean(step_uncertainty, axis=1)  # (n_samples,)
    
    return cumulative_uncertainty / horizon


def visualize_uncertainty_landscape(dyn_model, traj_obs, state_dims=(0, 1), 
                                     n_grid=50, target_states=None):
    """
    Visualize 2D uncertainty landscape for state space exploration.
    
    Parameters:
    -----------
    dyn_model : EnsembleSINDyDynamicsModel
        Trained ensemble model
    traj_obs : list of arrays
        Collected trajectory observations
    state_dims : tuple
        Which state dimensions to plot (default: x, y positions)
    n_grid : int
        Grid resolution
    target_states : np.array, optional
        High-uncertainty target states to overlay
    """
    import matplotlib.pyplot as plt
    
    all_obs = np.vstack([obs for obs in traj_obs])
    
    # Create grid over observed state space
    dim1, dim2 = state_dims
    x_min, x_max = all_obs[:, dim1].min(), all_obs[:, dim1].max()
    y_min, y_max = all_obs[:, dim2].min(), all_obs[:, dim2].max()
    
    x_range = np.linspace(x_min, x_max, n_grid)
    y_range = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Sample random actions
    random_actions = np.random.uniform(-1, 1, (n_grid * n_grid, 2))
    
    # Create grid states (set other dims to mean values)
    grid_states = np.tile(all_obs.mean(axis=0), (n_grid * n_grid, 1))
    grid_states[:, dim1] = X.flatten()
    grid_states[:, dim2] = Y.flatten()
    
    # Compute uncertainty at each grid point
    uncertainties = []
    for state, action in zip(grid_states, random_actions):
        ensemble_preds = []
        for idx in range(20):
            dyn_model.set_idx_coef_(idx)
            try:
                pred = dyn_model.predict(state, action)
                ensemble_preds.append(pred)
            except:
                ensemble_preds.append(np.full_like(state, np.nan))
        
        ensemble_preds = np.array(ensemble_preds)
        unc = np.nanmean(np.nanstd(ensemble_preds, axis=0))
        uncertainties.append(unc)
    
    uncertainties = np.array(uncertainties).reshape(n_grid, n_grid)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(X, Y, uncertainties, levels=20, cmap='YlOrRd')
    ax.scatter(all_obs[:, dim1], all_obs[:, dim2], c='blue', 
               s=1, alpha=0.1, label='Observed data')
    
    if target_states is not None:
        ax.scatter(target_states[:, dim1], target_states[:, dim2], 
                   c='lime', s=200, marker='*', edgecolors='black',
                   linewidths=2, label='Exploration targets', zorder=10)
    
    plt.colorbar(im, ax=ax, label='Prediction Uncertainty')
    ax.set_xlabel(f'State dimension {dim1}')
    ax.set_ylabel(f'State dimension {dim2}')
    ax.set_title('Ensemble Uncertainty Landscape')
    ax.legend()
    plt.tight_layout()
    plt.show()


# Example usage function
def active_learning_iteration(dyn_model, traj_obs, traj_acts, env, 
                                n_exploration_states=5):
    """
    Complete active learning iteration: find uncertain states and explore them.
    
    Returns:
    --------
    new_obs : list
        New observations from exploration
    new_acts : list  
        New actions from exploration
    target_states : np.array
        The states that were explored
    """
    # Find high-uncertainty states
    target_states, uncertainty_scores = find_high_uncertainty_states(
        dyn_model, traj_obs, traj_acts, 
        n_states=n_exploration_states,
        horizon=10
    )
    
    print(f"\nExploring {len(target_states)} high-uncertainty states with MPC...")
    
    # Visualize (optional)
    # visualize_uncertainty_landscape(dyn_model, traj_obs, target_states=target_states)
    
    return target_states, uncertainty_scores