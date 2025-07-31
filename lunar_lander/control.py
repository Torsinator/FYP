import os
import gymnasium as gym
import custom_lunar_lander
from gymnasium.wrappers import RecordVideo
import numpy as np
import do_mpc
import casadi as ca

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

    m = 5   # mass
    h = 14 / 30   # distance of thruster from COM
    a = 1   # side length
    I = 1 / 6 * m * a**2    # moment of inertia (assumes square)

    # Differential Equations
    model.set_rhs('x', vx)
    model.set_rhs('y', vy)
    model.set_rhs('vx', main_thrust * 13 * 2 / m  * ca.sin(theta) + side_thrust * 0.6 / m * ca.cos(theta))
    model.set_rhs('vy', side_thrust * 0.6 / m  * ca.sin(theta) + main_thrust * 13 * 2 / m * ca.cos(theta) - 10) # might be negative side thrust
    model.set_rhs('theta', omega)
    model.set_rhs('omega', -h / I * side_thrust * 0.6)

    model.setup()
    return model

def lunarify_target_state(state):
    x, y, vx, vy, theta, omega, *_ = state
    return np.array([x / 10, y / 6.666, vx / 5, vy / 7.5, theta, omega / 2.5])

model = create_lander_model()

# === Setup MPC Controller ===
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 20,
    't_step': 0.02,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

# Set target state here - will be LLM
target_state = np.array([0,5,0,0,0,0], dtype=np.float32)

mterm = (target_state[0] - model.x['x'])**2 + (target_state[1] - model.x['y'])**2 + (target_state[4] - model.x['theta'])**2
# mterm = (target_state[1] - model.x['y'])**2
lterm = mterm
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(main_thrust=0.01, side_thrust=0.01)

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

# Now, create a new environment for the final demonstration episode.
video_folder = "./final_video"
os.makedirs(video_folder, exist_ok=True)

# For video recording, "render_mode" must be set to "rgb_array".
demo_env = gym.make("CustomLunarLander-v0", render_mode="rgb_array", continuous=True)
# Wrap the environment so that it always records the (only) episode.
demo_env = RecordVideo(demo_env, video_folder=video_folder,
                        episode_trigger=lambda episode: True, name_prefix="MPC-controller")

# Run exactly one episode and record it.
state, info = demo_env.reset(options={"target_state" : lunarify_target_state(target_state)})
done = False

# Map Gym state to MPC state format
def gym_state_to_mpc(state):
    x, y, vx, vy, theta, omega, *_ = state
    # Multipliers from Lunar Lander
    return np.array([x * 10, y * 6.666, vx * 5, vy * 7.5, theta, omega * 2.5])

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
    # gym_action = np.array([0.15, 0])
    state, reward, terminated, truncated, info = demo_env.step(gym_action)
    done = terminated or truncated

demo_env.close()
print(f"Final demonstration video recorded and saved in: {video_folder}")

# fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
# fig.align_ylabels()

# mpc_graphics.reset_axes()
# Show the figure:
# fig