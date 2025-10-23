import numpy as np
import do_mpc
import casadi as ca
import environments.lunar_lander_customs as envs

def load_rhs_equations(file):
    with open(file, 'r') as file:
        string_list = [line.strip() for line in file]
    return string_list

# === Dynamics Model ===
def create_lander_model(rhs_exprs):
    model_type = 'discrete'
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

    for i in range(len(rhs_exprs)):
        model.set_rhs(f'x{i}', eval(rhs_exprs[i]))

    model.setup()
    return model


# Map Gym state to MPC state format
def gym_state_to_mpc(state):
    # print(f"TW: state {state}")
    if len(state) >= 6:
        x, y, vx, vy, theta, omega, *_ = state
        # Multipliers from Lunar Lander
        # print(f"TW: state to MPC {np.array([x * 10, y * 6.666, vx * 5, vy * 7.5, theta, omega * 2.5])}")
        return np.array([x * 10, y * 6.666, vx * 5, vy * 7.5, theta, omega * 2.5])
    else:
        x, y, theta, *_ = state
        return np.array([x * 10, y * 6.666, theta])

class SINDy():
    def __init__(self, gym_env, ode_file):
        self.env = gym_env
        self.last_target = np.array([])
        self.ode_file = ode_file
    
    @classmethod
    def load_model(cls):
        env = envs.LunarLanderMPC.make_env()
        model = SINDy(env, f"models/SINDy.txt")
        return model, env
    
    @classmethod
    def create_model(cls):
        return SINDy.load_model()
    
    def set_env(self, gym_env):
        self.env = gym_env
    
    def learn(self, total_timesteps):
        pass
    
    def save(self, path):
        pass

    def setup(self, obs):
        # === Setup MPC Controller ===
        print("MPC reset")
        model = create_lander_model(load_rhs_equations(self.ode_file))
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': 50,
            't_step': 0.02,
            'n_robust': 1,
            'store_full_solution': False,
        }
        mpc.set_param(**setup_mpc)
        mpc.settings.supress_ipopt_output()
        target_state = gym_state_to_mpc(self.env.target)
        weights = self.env.weights
        print("target:", target_state)
        print("weights:", weights)
        # mterm = (target_state[0] - model.x['x'])**2 + (target_state[1] - model.x['y'])**2 + (target_state[2] - model.x['vx'])**2 + (target_state[3] - model.x['vy'])**2 + (target_state[4] - model.x['theta'])**2 + (target_state[5] - model.x['omega'])**2
        mterm = weights[0] * (target_state[0] - model.x['x0'])**2 + weights[1] * (target_state[1] - model.x['x1'])**2 + weights[2] * (target_state[2] - model.x['x4'])**2
        # mterm = 0 * (target_state[0] - model.x['x0'])**2 + 0 * (target_state[1] - model.x['x1'])**2 + weights[2] * (target_state[2] - model.x['x4'])**2
        lterm = mterm
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u0=100, u1=100)

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
        mpc.x0 = gym_state_to_mpc(obs)
        mpc.set_initial_guess()
        self.mpc = mpc
        self.model = model

    def predict(self, obs, **kwargs) -> tuple:
        state = gym_state_to_mpc(obs)
        if not np.array_equal(self.env.target, self.last_target):
            self.setup(state)
        self.mpc.x0 = state
        self.mpc.set_initial_guess()
        action = self.mpc.make_step(state)
        main_thrust = float(action[0])
        side_thrust = float(action[1])
        self.last_target = self.env.target
        return np.array([main_thrust, side_thrust]), None
    