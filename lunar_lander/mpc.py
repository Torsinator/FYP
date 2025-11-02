import os
from Sindy_Env_Wrapper import LunarLanderSindy
from gymnasium.wrappers import RecordVideo
import numpy as np
import do_mpc
import casadi as ca
import matplotlib.pyplot as plt
from sindy_rl.policy import BasePolicy
from sindy_rl.env import safe_reset, safe_step
from tqdm import tqdm

class MPCPolicy(BasePolicy):
    dt = 0.1
    n_horizon = 20
    def __init__(self, library, coeffs, target=None):
        self.setup = False
        self.target = target
        self.model = create_lander_model(library, coeffs)
        if self.target is not None:
            self.mpc = setup_mpc(self.model, self.target, self.dt, self.n_horizon)

    def set_target(self, target):
        self.target = target
        self.mpc = setup_mpc(self.model, self.target, self.dt, self.n_horizon)
        self.setup = False

    def compute_action(self, obs):
        assert(self.target is not None)
        if not self.setup:
            self.mpc.x0 = obs
            self.mpc.set_initial_guess()
            self.setup = True
        return self.mpc.make_step(obs).flatten().astype(np.float32)
    
    def rollout_env(self, env, targets, seed=None, verbose = False, env_callback = None):
        trajs_obs = []
        trajs_acts = []
        trajs_rews = []

        for i in tqdm(range(len(targets)), disable=not verbose):
            k = 0
            target = targets[i]
            self.set_target(target)
            done = False
            if seed is not None:    
                obs_list = [safe_reset(env.reset(seed=seed))]
            else:
                obs_list = [safe_reset(env.reset())]
            act_list = []
            rew_list = []
            obs = obs_list[-1]
            while not done:
                # collect experience
                if k % 5 == 0:
                    k = 0
                    self.mpc.x0 = obs
                    self.mpc.set_initial_guess()
                    action = self.compute_action(obs)  # Get optimal control action
                print("action", action)
                obs, rew, done, info = safe_step(env.step(action))
                act_list.append(action)
                obs_list.append(obs)
                rew_list.append(rew)
                k += 1
                # print(f"obs: {obs_list}")
                # print(f"action: {act_list}")
                # exit()

            obs_list.pop(-1)
            trajs_obs.append(np.array(obs_list))
            trajs_acts.append(np.array(act_list))
            trajs_rews.append(np.array(rew_list))
            
            # env callback
            if env_callback:
                env_callback(i, env)
            assert(len(trajs_obs) == len(trajs_acts))
        return trajs_obs, trajs_acts, trajs_rews

def setup_mpc(model, target_state, dt, n_horizon):
    # === Setup MPC Controller ===
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': n_horizon,
        't_step': dt,
        'n_robust': 1,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    print(target_state)
    # mterm = (target_state[0] - model.x['x0'])**2 + (target_state[1] - model.x['x1'])**2 + (target_state[2] - model.x['x2'])**2 + (target_state[3] - model.x['x3'])**2 + (target_state[4] - model.x['x4'])**2 + (target_state[5] - model.x['x5'])**2
    mterm = (target_state[0] - model.x['x0'])**2 + (target_state[1] - model.x['x1'])**2 + (target_state[4] - model.x['x4'])**2
    lterm = mterm
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u0=1e-2, u1=1e-2)

    # Lower bounds on states:
    mpc.bounds['lower','_x', 'x0'] = -10
    mpc.bounds['lower','_x', 'x1'] = 0
    mpc.bounds['lower','_x', 'x2'] = -2*np.pi
    # Upper bounds on states
    mpc.bounds['upper','_x', 'x0'] = 10
    mpc.bounds['upper','_x', 'x1'] = 1.5 * 6.666
    mpc.bounds['upper','_x', 'x2'] = 2*np.pi

    # Lower bounds on inputs:
    mpc.bounds['lower','_u', 'u0'] = 0
    mpc.bounds['lower','_u', 'u1'] = -1
    # Upper bounds on inputs:
    mpc.bounds['upper','_u', 'u0'] = 1
    mpc.bounds['upper','_u', 'u1'] = 1

    mpc.setup()

    return mpc


# === Dynamics Model ===
def create_lander_model(library, coeffs):
    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)

    # States:
    x0 = model.set_variable('_x', 'x0', shape=(1,1))
    x1 = model.set_variable('_x', 'x1', shape=(1,1))
    x2 = model.set_variable('_x', 'x2', shape=(1,1))
    x3 = model.set_variable('_x', 'x3', shape=(1,1))
    x4 = model.set_variable('_x', 'x4', shape=(1,1))
    x5 = model.set_variable('_x', 'x5', shape=(1,1))

    # Controls: main engine thrust, side engine thrust
    u0 = model.set_variable('_u', 'u0')
    u1 = model.set_variable('_u', 'u1')

    rhs_exprs = []
    for term in coeffs:
        expr_str = ""
        for i in range(len(term)):
            if float(term[i]) > 0.1:
                if len(expr_str) > 0:
                    expr_str += "+ "
                expr_str += f"{term[i]}*{library[i]}"
        rhs_exprs.append(expr_str)

    for i in range(len(rhs_exprs)):
        print(rhs_exprs[i])
        model.set_rhs(f'x{i}', eval(rhs_exprs[i]))

    model.setup()
    return model

def lunarify_target_state(state):
    x, y, vx, vy, theta, omega, *_ = state
    return np.array([x / 10, y / 6.666, vx / 5, vy / 7.5, theta, omega / 2.5])

def run_episode(model, states):
    # Now, create a new environment for the final demonstration episode.
    video_folder = "./final_video"
    os.makedirs(video_folder, exist_ok=True)

    # For video recording, "render_mode" must be set to "rgb_array".
    demo_env = LunarLanderSindy.make_env()
    # Wrap the environment so that it always records the (only) episode.
    demo_env = RecordVideo(demo_env, video_folder=video_folder,
                            episode_trigger=lambda episode: True, name_prefix="MPC-controller")

    # Run exactly one episode and record it.
    # state, info = demo_env.reset(options={"target_state" : lunarify_target_state(target_state)})
    state, info = demo_env.reset(options={"target_state" : states[0]})

    for target_state in states:
        k = 0
        done = False
        demo_env.unwrapped.set_target_state(np.array(target_state, dtype=np.float32))
        mpc = setup_mpc(model, target_state, 0.02, 120)
        # === Control Loop ===
        while not done:
            print(state)
            print("State input to MPC:", state)
            if k % 5 == 0:
                k = 0
                mpc.x0 = state
                mpc.set_initial_guess()
                action = mpc.make_step(state)  # Get optimal control action
            main_thrust = float(action[0])
            side_thrust = float(action[1])
            print(main_thrust, side_thrust)

            # Apply to Gym simulator
            gym_action = np.array([main_thrust, side_thrust])
            # gym_action = np.array([0, -1])
            state, reward, terminated, truncated, info = demo_env.step(gym_action)
            done = terminated or truncated or abs(reward) > 999
            k += 1

    demo_env.close()
    print(f"Final demonstration video recorded and saved in: {video_folder}")