from agents.Agent import Agent
import numpy as np
import do_mpc
import casadi as ca

MAIN_POWER = 100
SIDE_POWER = 20
GRAVITY = -10
dt = 0.02

def main_thrust_fn(main_thrust):
    # thrust = (0.5 * ca.tanh(10*(2* main_thrust - 1))+ 0.5)*(0.5 * ca.fmax(0, 2*main_thrust - 1) + 0.5)
    return main_thrust * MAIN_POWER

def smooth_step(x, threshold=0.5, sharpness=20):
    abs_x = ca.sqrt(x**2)  # Smooth abs
    return 1 / (1 + ca.exp(-sharpness * (abs_x - threshold)))

def side_thrust_fn(side_thrust):
    # activation = smooth_step(side_thrust, threshold=0.5, sharpness=20)
    return side_thrust * SIDE_POWER

# === Dynamics Model ===
def create_lander_model():
    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)

    # States:
    x = model.set_variable('_x', 'x', shape=(1,1))
    y = model.set_variable('_x', 'y', shape=(1,1))
    vx = model.set_variable('_x', 'vx', shape=(1,1))
    vy = model.set_variable('_x', 'vy', shape=(1,1))
    theta = model.set_variable('_x', 'theta', shape=(1,1))
    omega = model.set_variable('_x', 'omega', shape=(1,1))

    # Controls: main engine thrust, side engine thrust
    main_thrust = model.set_variable('_u', 'main_thrust')
    side_thrust = model.set_variable('_u', 'side_thrust')

    m = 4.816666603088379   # mass
    h = 14.0 / 30   # distance of thruster from COM
    a = 1   # side length
    # I = 1 / 6 * m * a**2    # moment of inertia (assumes square)
    I = 0.8333148956298828

    # Differential Equations
    model.set_rhs('x', vx)
    model.set_rhs('y', vy)
    F_body_x = side_thrust_fn(side_thrust)
    F_body_y = main_thrust_fn(main_thrust)

    F_world_x = ca.cos(theta) * F_body_x - ca.sin(theta) * F_body_y
    F_world_y = ca.sin(theta) * F_body_x + ca.cos(theta) * F_body_y

    model.set_rhs('vx', F_world_x / m)
    model.set_rhs('vy', F_world_y / m + GRAVITY)  # assuming g = 10 m/s²
    model.set_rhs('theta', omega)
    model.set_rhs('omega', -h / I * side_thrust_fn(side_thrust))

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

class MPCAgent(Agent):
    def __init__(self, gym_env):
        self.env = gym_env
        self.last_target = np.array([])

    @staticmethod
    def load(model_path, gym_env=None) -> Agent:
        return MPCAgent(gym_env)
    
    def set_env(self, gym_env):
        self.env = gym_env
    
    def learn(self, total_timesteps):
        pass
    
    def save(self, path):
        pass

    def setup(self, obs):
        # === Setup MPC Controller ===
        model = create_lander_model()
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': 120,
            't_step': dt,
            'n_robust': 1,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)
        target_state = gym_state_to_mpc(self.env.env.target_state)
        self.last_target = target_state
        # mterm = (target_state[0] - model.x['x'])**2 + (target_state[1] - model.x['y'])**2 + (target_state[2] - model.x['vx'])**2 + (target_state[3] - model.x['vy'])**2 + (target_state[4] - model.x['theta'])**2 + (target_state[5] - model.x['omega'])**2
        mterm = (target_state[0] - model.x['x'])**2 + (target_state[1] - model.x['y'])**2 + (target_state[2] - model.x['theta'])**2
        lterm = mterm
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(main_thrust=100, side_thrust=100)

        # Lower bounds on states:
        mpc.bounds['lower','_x', 'x'] = -10
        mpc.bounds['lower','_x', 'y'] = 0
        mpc.bounds['lower','_x', 'theta'] = -2*np.pi
        # Upper bounds on states
        mpc.bounds['upper','_x', 'x'] = 10
        mpc.bounds['upper','_x', 'y'] = 1.5 * 6.666
        mpc.bounds['upper','_x', 'theta'] = 2*np.pi

        # Lower bounds on inputs:
        mpc.bounds['lower','_u', 'main_thrust'] = 0
        mpc.bounds['lower','_u', 'side_thrust'] = -1
        # Upper bounds on inputs:
        mpc.bounds['upper','_u', 'main_thrust'] = 1
        mpc.bounds['upper','_u', 'side_thrust'] = 1

        mpc.setup()
        mpc.x0 = obs
        mpc.set_initial_guess()
        self.mpc = mpc
        self.model = model

    def predict(self, obs, **kwargs) -> tuple:
        print(f"TW obs: {obs["observation"]}")
        state = gym_state_to_mpc(obs["observation"])
        if not np.array_equal(gym_state_to_mpc(obs["desired_goal"]), self.last_target):
            self.setup(state)
        action = self.mpc.make_step(state)
        main_thrust = float(action[0])
        side_thrust = float(action[1])
        return np.array([main_thrust, side_thrust]), None
    