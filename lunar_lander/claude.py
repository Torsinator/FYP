import numpy as np
import do_mpc
from casadi import *
from sindy_rl.dynamics import EnsembleSINDyDynamicsModel
from Sindy_Env_Wrapper import LunarLanderSindy
from sindy_rl.env import rollout_env
from sindy_rl.policy import RandomPolicy
from matplotlib import pyplot as plt
from tqdm import tqdm

# ==================== Configuration ====================
dyna_config = {
    'dt': 0.02,
    'discrete': True,
    'optimizer': {
        'base_optimizer': {
            'name': 'STLSQ',
            'kwargs': {
                'alpha': 5.0e-5,
                'threshold': 5e-7,
            },
        },
        'ensemble': {
            'bagging': True,
            'library_ensemble': True,
            'n_models': 20,
        },
    },
    'feature_library': {
        'name': 'affine',
        'kwargs': {
            'poly_deg': 2,
            'n_state': 6,
            'n_control': 2,
            'poly_int': True,
            'tensor': True,
        }
    }
}

N_ITERATIONS = 5  # Number of active learning iterations
INITIAL_DATA_STEPS = 8000
NEW_DATA_STEPS = 1000
MPC_HORIZON = 20

# ==================== Helper Functions ====================
def compute_model_uncertainty(dyn_model, states, actions):
    """Compute prediction variance across ensemble for given state-action pairs"""
    predictions = []
    for idx in range(20):  # assuming 20 ensemble models
        dyn_model.set_idx_coef_(idx)
        preds = []
        for s, a in zip(states, actions):
            try:
                pred = dyn_model.predict(s, a)
                preds.append(pred)
            except:
                preds.append(np.full_like(s, np.nan))
        predictions.append(np.array(preds))
    
    predictions = np.array(predictions)  # shape: (n_models, n_samples, n_states)
    uncertainty = np.nanstd(predictions, axis=0)  # std across ensemble
    return np.mean(uncertainty, axis=1)  # average uncertainty per sample


def build_casadi_dynamics_from_sindy(dyn_model, x_sym, u_sym):
    """Convert SINDy polynomial model to CasADi symbolic expression"""
    from casadi import SX, vertcat as sx_vertcat
    
    # Get the coefficient list from the SINDy model
    coef_list = dyn_model.get_coef_list()  # Returns list of coefficient matrices
    
    # Use median coefficients
    dyn_model.set_median_coef_()
    coefs = np.array(coef_list[10])  # Median is typically at index 10 for 20 models
    
    # Get feature library to understand the polynomial structure
    n_state = 6
    n_control = 2
    
    # Build feature vector symbolically [1, x, u, x^2, x*u, u^2, ...]
    # This depends on your affine library config (poly_deg=2, tensor=True)
    features = []
    
    # Constant term
    features.append(1.0)
    
    # Linear terms: x and u
    for i in range(n_state):
        features.append(x_sym[i])
    for i in range(n_control):
        features.append(u_sym[i])
    
    # Quadratic terms: x^2, x*u, u^2 (with tensor=True)
    for i in range(n_state):
        for j in range(i, n_state):
            features.append(x_sym[i] * x_sym[j])
    
    for i in range(n_state):
        for j in range(n_control):
            features.append(x_sym[i] * u_sym[j])
    
    for i in range(n_control):
        for j in range(i, n_control):
            features.append(u_sym[i] * u_sym[j])
    
    # Build dynamics: x_dot = sum of (coef * feature) for each state dimension
    # Use SX.zeros to create symbolic result
    x_dot = SX.zeros(n_state, 1)
    
    for i in range(n_state):
        for j in range(min(len(features), coefs.shape[0])):
            coef_val = float(coefs[j, i])
            if abs(coef_val) > 1e-10:  # Skip near-zero coefficients
                x_dot[i] += coef_val * features[j]
    
    return x_dot


def create_mpc_controller(dyn_model, dt=0.02, n_horizon=20):
    """Create MPC controller using median SINDy model"""
    from casadi import SX
    
    dyn_model.set_median_coef_()
    
    # Setup MPC model
    model_type = 'continuous'  # SINDy learns continuous dynamics (dx/dt)
    model = do_mpc.model.Model(model_type)
    
    # States - use SX explicitly
    x = model.set_variable('_x', 'x', shape=(6,1))
    
    # Controls - use SX explicitly
    u = model.set_variable('_u', 'u', shape=(2,1))
    
    # Verify we have SX types
    print(f"  x type: {type(x)}, u type: {type(u)}")
    
    # Build CasADi dynamics from SINDy coefficients
    try:
        x_dot = build_casadi_dynamics_from_sindy(dyn_model, x, u)
        print(f"  x_dot type: {type(x_dot)}")
    except Exception as e:
        print(f"Warning: Could not build symbolic dynamics from SINDy: {e}")
        exit()
    
    model.set_rhs('x', x_dot)
    model.setup()
    
    # Setup MPC
    mpc = do_mpc.controller.MPC(model)
    
    setup_mpc = {
        'n_horizon': n_horizon,
        't_step': dt,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    
    # Objective: minimize state deviation (can be modified for exploration)
    mterm = sum1(x**2)
    lterm = sum1(x**2)
    
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=0.1)  # Control cost
    
    # Constraints
    mpc.bounds['lower','_u','u'] = -1.0
    mpc.bounds['upper','_u','u'] = 1.0
    
    mpc.setup()
    
    return mpc


class UncertaintyExplorationPolicy:
    """Policy that uses MPC to drive system to high-uncertainty regions"""
    def __init__(self, mpc_controller, dyn_model, uncertainty_map=None):
        self.mpc = mpc_controller
        self.dyn_model = dyn_model
        self.uncertainty_map = uncertainty_map
        self.step_count = 0
        
    def __call__(self, obs):
        self.step_count += 1
        # Use MPC to generate action
        try:
            # Set MPC state
            self.mpc.x0 = obs.reshape(-1, 1)
            self.mpc.set_initial_guess()
            
            # Compute control action
            u = self.mpc.make_step(obs.reshape(-1, 1))
            return u.flatten()
        except Exception as e:
            if self.step_count < 5:  # Only print first few errors
                print(f"  MPC failed: {e}, using random action")
            # Fallback to random action if MPC fails
            return np.random.uniform(-1, 1, size=2)


# ==================== Main Active Learning Loop ====================
def active_learning_loop():
    print("=" * 60)
    print("Active Learning SINDy-RL with MPC")
    print("=" * 60)
    
    # Initialize environment
    env = LunarLanderSindy.make_env()
    
    # Step 1: Initial random data collection
    print("\n[Iteration 0] Collecting initial random data...")
    random_policy = RandomPolicy(env.action_space)
    all_obs, all_acts, all_rews = rollout_env(
        env, random_policy, 
        n_steps=INITIAL_DATA_STEPS, 
        n_steps_reset=1000
    )
    
    # Initialize dynamics model
    dyn_model = EnsembleSINDyDynamicsModel(dyna_config)
    
    iteration_metrics = []
    
    for iteration in range(N_ITERATIONS):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}/{N_ITERATIONS}")
        print(f"{'=' * 60}")
        
        # Step 2: Fit SINDy model on current data
        print(f"[Step 1] Training SINDy ensemble on {len(all_obs[-1])} data points...")
        train_obs = all_obs[:-1] if len(all_obs) > 1 else all_obs
        train_acts = all_acts[:-1] if len(all_acts) > 1 else all_acts
        
        dyn_model.fit(train_obs, train_acts)
        dyn_model.set_median_coef_()
        
        # Step 3: Evaluate model uncertainty
        print("[Step 2] Computing model uncertainty...")
        test_obs = all_obs[-1][:500]  # Sample for uncertainty estimation
        test_acts = all_acts[-1][:500]
        uncertainties = compute_model_uncertainty(dyn_model, test_obs, test_acts)
        
        avg_uncertainty = np.mean(uncertainties)
        max_uncertainty = np.max(uncertainties)
        
        print(f"  → Avg uncertainty: {avg_uncertainty:.4f}")
        print(f"  → Max uncertainty: {max_uncertainty:.4f}")
        
        iteration_metrics.append({
            'iteration': iteration,
            'avg_uncertainty': avg_uncertainty,
            'max_uncertainty': max_uncertainty,
            'n_data': sum(len(o) for o in all_obs)
        })
        
        # Step 4: Create MPC controller with current model
        print("[Step 3] Setting up MPC controller...")
        # Note: Simplified MPC for demo - full implementation needs SINDy→CasADi conversion
        mpc = create_mpc_controller(dyn_model, dt=dyna_config['dt'], n_horizon=MPC_HORIZON)
        
        # Step 5: Use MPC to explore uncertain regions
        print("[Step 4] Collecting new data using MPC-guided exploration...")
        exploration_policy = UncertaintyExplorationPolicy(mpc, dyn_model, uncertainties)
        
        # Use MPC-based exploration
        new_obs, new_acts, new_rews = rollout_env(
            env, exploration_policy, 
            n_steps=NEW_DATA_STEPS,
            n_steps_reset=500
        )
        
        # Step 6: Aggregate new data
        print("[Step 5] Updating dataset...")
        all_obs.extend(new_obs)
        all_acts.extend(new_acts)
        all_rews.extend(new_rews)
        
        print(f"  → Total trajectories: {len(all_obs)}")
        print(f"  → Total data points: {sum(len(o) for o in all_obs)}")
    
    print(f"\n{'=' * 60}")
    print("Active Learning Complete!")
    print(f"{'=' * 60}")
    
    # Final model evaluation
    print("\n[Final Model]")
    dyn_model.print()
    
    # Plot learning curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    iterations = [m['iteration'] for m in iteration_metrics]
    avg_uncs = [m['avg_uncertainty'] for m in iteration_metrics]
    max_uncs = [m['max_uncertainty'] for m in iteration_metrics]
    
    axes[0].plot(iterations, avg_uncs, 'o-', label='Avg Uncertainty')
    axes[0].plot(iterations, max_uncs, 's-', label='Max Uncertainty')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Uncertainty')
    axes[0].set_title('Model Uncertainty vs Iteration')
    axes[0].legend()
    axes[0].grid(True)
    
    n_data = [m['n_data'] for m in iteration_metrics]
    axes[1].plot(iterations, n_data, 'o-', color='green')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Total Data Points')
    axes[1].set_title('Data Collection Progress')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return dyn_model, all_obs, all_acts, iteration_metrics


# ==================== Run ====================
if __name__ == "__main__":
    final_model, observations, actions, metrics = active_learning_loop()