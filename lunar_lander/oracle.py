import numpy as np
from scipy.integrate import solve_ivp


# -------------------------
# Physical constants
# -------------------------
MAIN_POWER = 100.0
SIDE_POWER = 20.0
GRAVITY = -10.0      # m/s² (downward)
m = 4.816666603088379
h = 14.0 / 30.0
I = 0.8333148956298828
dt = 0.02


# -------------------------
# True lunar lander ODEs
# -------------------------
def lander_ode(t, s, u_func):
    """
    s = [x0, x1, x2, x3, x4, x5]
        x0 = horizontal position
        x1 = vertical position
        x2 = horizontal velocity
        x3 = vertical velocity
        x4 = angle (theta)
        x5 = angular velocity
    u_func(t) -> (u0, u1)
        u0 = main engine throttle (0–1)
        u1 = side thruster throttle (–1–1)
    """
    x0, x1, x2, x3, x4, x5 = s
    u0, u1 = u_func(t)

    # Forces in world frame
    F_world_x = (np.cos(x4) * SIDE_POWER * u1 - np.sin(x4) * MAIN_POWER * u0) / m
    F_world_y = (np.sin(x4) * SIDE_POWER * u1 + np.cos(x4) * MAIN_POWER * u0) / m

    # Derivatives
    dx0dt = x2
    dx1dt = x3
    dx2dt = F_world_x
    dx3dt = F_world_y + GRAVITY      # include downward acceleration
    dx4dt = x5
    dx5dt = -h / I * SIDE_POWER * u1

    return [dx0dt, dx1dt, dx2dt, dx3dt, dx4dt, dx5dt]


# -------------------------
# Control input generator
# -------------------------
def random_u(t, mode="smooth"):
    """Random but smooth or piecewise constant control signal."""
    # You can adjust frequencies and amplitudes for excitation
    if mode == "smooth":
        u0 = 0.5 + 0.3 * np.sin(0.4 * t) + 0.2 * np.random.randn() * 0.1
        u1 = 0.8 * np.sin(0.7 * t + 1.2)
    elif mode == "impulse":
        u0 = np.random.choice([0.0, 0.5, 1.0])
        u1 = np.random.uniform(-1.0, 1.0)
    else:
        u0 = np.random.uniform(0, 1)
        u1 = np.random.uniform(-1, 1)
    return np.clip(u0, 0.0, 1.0), np.clip(u1, -1.0, 1.0)


# -------------------------
# Trajectory generation
# -------------------------
def generate_trajectory(t_span=(0, 10), dt=dt, mode="smooth", s0=None):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    if s0 is None:
        s0 = np.array([
            np.random.uniform(-2, 2),    # x0
            np.random.uniform(5, 10),    # x1
            np.random.uniform(-1, 1),    # x2
            np.random.uniform(-1, 1),    # x3
            np.random.uniform(-0.3, 0.3),# x4 (radians)
            np.random.uniform(-0.1, 0.1) # x5
        ])

    sol = solve_ivp(
        lambda t, s: lander_ode(t, s, lambda t_: random_u(t_, mode)),
        t_span, s0, t_eval=t_eval, rtol=1e-8, atol=1e-8
    )

    X = sol.y.T
    U = np.array([random_u(t, mode) for t in sol.t])
    return sol.t, X, U


# -------------------------
# Batch data generation
# -------------------------
def generate_dataset(n_traj=100, t_span=(0, 10), dt=dt, mode="smooth"):
    all_X, all_U = [], []
    for _ in range(n_traj):
        _, X, U = generate_trajectory(t_span, dt, mode)
        all_X.append(X)
        all_U.append(U)
    X = np.concatenate(all_X, axis=0)
    U = np.concatenate(all_U, axis=0)
    return X, U


if __name__ == "__main__":
    t, X, U = generate_trajectory()
    print("Trajectory shape:", X.shape, U.shape)

    X_all, U_all = generate_dataset(n_traj=50)
    print("Dataset:", X_all.shape, U_all.shape)

    # Example plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X[:, 0], X[:, 1])
    plt.xlabel("x0 (horizontal)")
    plt.ylabel("x1 (vertical)")
    plt.title("Example synthetic lunar lander trajectory")
    plt.show()
