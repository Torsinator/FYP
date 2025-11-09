from agents.Agent import Agent
import numpy as np
import do_mpc
import casadi as ca

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

    # Time varying parameters for weights and target
    target = model.set_variable('_tvp', 'target', shape=(3, 1))
    weights = model.set_variable('_tvp', 'weights', shape=(3, 1))

    print("registered tvps")

    for i in range(len(rhs_exprs)):
        print(rhs_exprs[i])
        model.set_rhs(f'x{i}', eval(rhs_exprs[i]))

    model.setup()
    return model


# Map Gym state to MPC state format
def gym_state_to_mpc(state):
    print(f"TW: state {state}")
    if len(state) == 9:
        x, y, theta, wx, wy, wz, vx, vy, omega, *_ = state
        # Multipliers from Lunar Lander
        print(f"TW: state to MPC {np.array([x * 10, y * 6.666, vx * 5, vy * 7.5, theta, omega * 2.5])}")
        return np.array([x * 10, y * 6.666, vx * 5, vy * 7.5, theta, omega * 2.5])
    else:
        x, y, theta, *_ = state
        return np.array([x * 10, y * 6.666, theta])

class ShapedSINDyAgent(Agent):
    def __init__(self, gym_env, ode_file):
        self.env = gym_env
        self.last_target = np.array([])
        self.last_weights = np.array([])
        self.ode_file = ode_file
        self.is_setup = False

    @staticmethod
    def load(model_path, gym_env=None) -> Agent:
        return ShapedSINDyAgent(gym_env, model_path)
    
    def set_env(self, gym_env):
        self.env = gym_env
    
    def learn(self, total_timesteps):
        pass
    
    def save(self, path):
        pass

    def setup(self, obs):
        # === Setup MPC Controller ===
        model = create_lander_model(load_rhs_equations(self.ode_file))
        mpc = do_mpc.controller.MPC(model)
        n_horizon = 60
        setup_mpc = {
            'n_horizon': n_horizon,
            't_step': 0.02,
            'n_robust': 1,
            'store_full_solution': False
        }
        mpc.set_param(**setup_mpc)
        # mterm = (target_state[0] - model.x['x'])**2 + (target_state[1] - model.x['y'])**2 + (target_state[2] - model.x['vx'])**2 + (target_state[3] - model.x['vy'])**2 + (target_state[4] - model.x['theta'])**2 + (target_state[5] - model.x['omega'])**2
        # Target TVPs
        err = ca.vertcat(model.x['x0'], model.x['x1'], model.x['x4']) - model.tvp['target'] 
        lterm = ca.dot(model.tvp['weights'], err**2) # + model.x['x1']**2 + model.x['x2']**2
        mterm = ca.SX(0)
        # lterm = ca.SX(0)
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u0=1e-4, u1=1e-4)

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

        def tvp_fun(t_now):
            tvp_template = mpc.get_tvp_template()
            for k in range(n_horizon + 1):
                tvp_template['_tvp', k, 'target'] = self.last_target
                tvp_template['_tvp', k, 'weights'] = self.last_weights
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        mpc.prepare_nlp()

        N = mpc.settings.n_horizon
        # target_state = ca.DM(self.last_target)
        # weights = ca.DM(self.last_weights)

        p = 10.0          # smoothness
        epsilon = 1e-6  # numerical guard
        reward_weight = N

        distances = []
        for k in range(N+1):
            xk = ca.vertcat(*mpc.opt_x['_x', k, 0])
            err = (xk[[0,1,4]] - mpc.opt_p['_tvp', k, 'target'])
            dist_sq = ca.dot(mpc.opt_p['_tvp', k, 'weights'], err**2)
            distances.append(dist_sq)

        dists = ca.vertcat(*distances)
        soft_min = (ca.sum1((dists + epsilon)**(-p)))**(-1/p)

        mpc.nlp_obj += reward_weight * soft_min
        mpc.create_nlp()
        mpc.x0 = obs
        mpc.set_initial_guess()
        self.mpc = mpc
        self.model = model
        mpc.compile_nlp(compiler_command="gcc -fPIC -shared -O3 {cname} -o {libname}".format(cname="nlp.c", libname="nlp.so"))
        self.is_setup = True

    def predict(self, obs, **kwargs) -> tuple:
        print(f"TW obs: {obs["observation"]}")
        self.last_weights = self.env.env.weights
        self.last_target = gym_state_to_mpc(self.env.env.target_state)
        state = gym_state_to_mpc(obs["observation"])
        print("current state", state)
        print("target state", self.last_target)
        if not self.is_setup:
            self.setup(state)
        action = self.mpc.make_step(state)
        main_thrust = float(action[0])
        side_thrust = float(action[1])
        return np.array([main_thrust, side_thrust]), None
    