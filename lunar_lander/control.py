import os
import gymnasium as gym
import custom_lunar_lander
from gymnasium.wrappers import RecordVideo
import numpy as np
import do_mpc
import casadi as ca
import matplotlib.pyplot as plt

MAIN_POWER = 100
SIDE_POWER = 20
GRAVITY = -10
dt = 0.02

def setup_mpc(model, target_state):
    # === Setup MPC Controller ===
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': 120,
        't_step': dt,
        'n_robust': 1,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    # mterm = (target_state[0] - model.x['x'])**2 + (target_state[1] - model.x['y'])**2 + (target_state[2] - model.x['vx'])**2 + (target_state[3] - model.x['vy'])**2 + (target_state[4] - model.x['theta'])**2 + (target_state[5] - model.x['omega'])**2
    mterm = (target_state[0] - model.x['x'])**2 + (target_state[1] - model.x['y'])**2 + (target_state[4] - model.x['theta'])**2
    lterm = mterm
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(main_thrust=100, side_thrust=100)

    # Lower bounds on states:
    mpc.bounds['lower','_x', 'x'] = -10
    mpc.bounds['lower','_x', 'y'] = 0
    mpc.bounds['lower','_x', 'theta'] = -np.pi
    # Upper bounds on states
    mpc.bounds['upper','_x', 'x'] = 10
    mpc.bounds['upper','_x', 'y'] = 1.5 * 6.666
    mpc.bounds['upper','_x', 'theta'] = np.pi

    # Lower bounds on inputs:
    mpc.bounds['lower','_u', 'main_thrust'] = 0
    mpc.bounds['lower','_u', 'side_thrust'] = -1
    # Upper bounds on inputs:
    mpc.bounds['upper','_u', 'main_thrust'] = 1
    mpc.bounds['upper','_u', 'side_thrust'] = 1

    mpc.setup()

    return mpc

def main_thrust_fn(main_thrust):
    # thrust = (0.5 * ca.tanh(10*(2* main_thrust - 1))+ 0.5)*(0.5 * ca.fmax(0, 2*main_thrust - 1) + 0.5)
    return main_thrust * MAIN_POWER

def smooth_step(x, threshold=0.5, sharpness=20):
    abs_x = ca.sqrt(x**2)  # Smooth abs
    return 1 / (1 + ca.exp(-sharpness * (abs_x - threshold)))


def side_thrust_fn(side_thrust):
    # activation = smooth_step(side_thrust, threshold=0.5, sharpness=20)
    return side_thrust * SIDE_POWER

x = np.linspace(-2, 2 ,100)
y = main_thrust_fn(x)
plt.plot(x, y)
plt.show()

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

def lunarify_target_state(state):
    x, y, vx, vy, theta, omega, *_ = state
    return np.array([x / 10, y / 6.666, vx / 5, vy / 7.5, theta, omega / 2.5])

model = create_lander_model()

states = np.array(
  [
  [-0.70000, 0.8, 0, 0, 0, 0],
  [-0.65000, 0.5, 0.20000, -6.28319, -0, 0.2],
  [-0.60000, 0.2, 0.20000, 0.2, 0, 10.00000],
  [-0.55000, 0.5, 0.20000, 6.28319, 0, 0.2],
  [-0.50000, 0.8, 0.20000, 0.2, 0, -10.00000],
  [-0.45000, 0.5, 0.20000, -6.28319, -0, 0.2],
  [-0.40000, 0.2, 0.20000, 0.2, 0, 10.00000],
  [-0.35000, 0.5, 0.20000, 6.28319, 0, 0.2],
  [-0.30000, 0.8, 0.20000, 0.2, 0, -10.00000],
  [-0.25000, 0.5, 0.20000, -6.28319, -0, 0.2],
  [-0.20000, 0.2, 0.20000, 0.2, 0, 10.00000],
  [-0.15000, 0.5, 0.20000, 6.28319, 0, 0.2],
  [-0.10000, 0.8, 0.20000, 0.2, 0, -10.00000],
  [-0.05000, 0.5, 0.20000, -6.28319, -0, 0.2],
  [0.00000, 0.2, 0.20000, 0.2, 0, 10.00000],
  [0.05000, 0.5, 0.20000, 6.28319, 0, 0.2],
  [0.10000, 0.8, 0.20000, 0.2, 0, -10.00000],
  [0.15000, 0.5, 0.20000, -6.28319, -0, 0.2],
  [0.20000, 0.2, 0.20000, 0.2, 0, 10.00000],
  [0.25000, 0.5, 0.20000, 6.28319, 0, 0.2],
  [0.30000, 0.8, 0.20000, 0.2, 0, -10.00000],
  [0.35000, 0.5, 0.20000, -6.28319, -0, 0.2],
  [0.40000, 0.2, 0.20000, 0.2, 0, 10.00000],
  [0.45000, 0.5, 0.20000, 6.28319, 0, 0.2],
  [0.50000, 0.8, 0.20000, 0.2, 0, -10.00000],
  [0.55000, 0.5, 0.20000, -6.28319, -0, 0.2],
  [0.60000, 0.2, 0.20000, 0.2, 0, 10.00000],
  [0.65000, 0.5, 0.20000, 6.28319, 0, 0.2],
  [0.70000, 0.8, 0.20000, 0.2, 0, -10.00000]
  ]
)

# Now, create a new environment for the final demonstration episode.
video_folder = "./final_video"
os.makedirs(video_folder, exist_ok=True)

# For video recording, "render_mode" must be set to "rgb_array".
demo_env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
# Wrap the environment so that it always records the (only) episode.
demo_env = RecordVideo(demo_env, video_folder=video_folder,
                        episode_trigger=lambda episode: True, name_prefix="MPC-controller")

# Run exactly one episode and record it.
# state, info = demo_env.reset(options={"target_state" : lunarify_target_state(target_state)})
state, info = demo_env.reset(options={"target_state" : states[0]})


# Map Gym state to MPC state format
def gym_state_to_mpc(state):
    x, y, vx, vy, theta, omega, *_ = state
    # Multipliers from Lunar Lander
    return np.array([x * 10, y * 6.666, vx * 5, vy * 7.5, theta, omega * 2.5])

for target_state in states:
    done = False
    demo_env.unwrapped.set_target_state(np.array(target_state, dtype=np.float32))
    mpc = setup_mpc(model, gym_state_to_mpc(target_state))
    mpc.x0 = gym_state_to_mpc(state)
    mpc.set_initial_guess()
    # === Control Loop ===
    while not done:
        print("State input to MPC:", gym_state_to_mpc(state[6:]))
        action = mpc.make_step(gym_state_to_mpc(state[6:]))  # Get optimal control action
        main_thrust = float(action[0])
        side_thrust = float(action[1])
        print(main_thrust, side_thrust)

        # Apply to Gym simulator
        gym_action = np.array([main_thrust, side_thrust])
        # gym_action = np.array([0, -1])
        state, reward, terminated, truncated, info = demo_env.step(gym_action)
        done = terminated or truncated or abs(reward) > 999

demo_env.close()
print(f"Final demonstration video recorded and saved in: {video_folder}")

# from matplotlib import rcParams
# rcParams['axes.grid'] = True
# rcParams['font.size'] = 18

# import matplotlib.pyplot as plt
# fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(16,9))
# graphics.plot_results()
# graphics.reset_axes()
# plt.show()

# fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
# fig.align_ylabels()

# mpc_graphics.reset_axes()
# Show the figure:
# fig