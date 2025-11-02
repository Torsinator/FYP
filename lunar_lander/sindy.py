import numpy as np
from pysindy.pysindy import solve_ivp
from pysindy.utils import lorenz_control
import pysindy as ps
import custom_lunar_lander_no_target
import gymnasium as gym
from coupled_fourier_library import CoupledFourierLibrary

# Map Gym state to MPC state format
def gym_state_to_mpc(state):
    x, y, vx, vy, theta, omega, *_ = state
    # Multipliers from Lunar Lander
    return np.array([x * 10, y * 6.666, vx * 5, vy * 7.5, theta, omega * 2.5])

# control input function
def u_fun(t):
    return np.array([np.max(np.sin(10*t), 0), np.sin(10*t)], dtype=np.float32)

dt = 1/50

combined_library = ps.GeneralizedLibrary([
    ps.PolynomialLibrary(degree=1),
    # CoupledFourierLibrary(2)
])

# Generate measurement data
env = gym.make('CustomLunarLander-v0', continuous=True)
episodes = 500
x_list = []
u_list = []

for ep in range(episodes):
    done = False
    t = 0
    x_ep, u_ep = [], []
    obs, _ = env.reset()
    state = gym_state_to_mpc(obs)
    x_ep.append(state)
    # u_ep.append([0,0])
    while not done:
        action = u_fun(t)
        obs, _, done, term, _ = env.step(action)
        state = gym_state_to_mpc(obs)
        done = done or term
        if not done:
            u_ep.append(action)
            x_ep.append(state)
        t += dt
    x_ep.pop()
    x_list.append(np.array(x_ep))
    u_list.append(np.array(u_ep))

model = ps.SINDy(
    feature_library=combined_library,
    optimizer=ps.STLSQ(threshold=0.2),
    differentiation_method=ps.SmoothedFiniteDifference()
)
model.fit(x_list, t=dt)
model.print()