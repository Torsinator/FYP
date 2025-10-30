import casadi as ca
import numpy as np
import time

# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
n = 20          # number of elements
eps = 1e-6
p_values = [1, 5, 10, 20, 50]

# Symbolic variables
dists = ca.SX.sym("d", n)

# Data for evaluation
d_test = np.linspace(0.1, 2, n)

# ----------------------------------------------------
# Benchmark differentiation time
# ----------------------------------------------------
print(f"{'p':>4} | {'grad_time (ms)':>14} | {'hess_time (ms)':>14} | {'grad_norm':>10} | {'softmin':>10}")
print("-" * 60)

for p in p_values:
    # Define the soft-min
    softmin_expr = (ca.sum1((dists + eps) ** (-p))) ** (-1 / p)

    # Measure differentiation time
    t0 = time.time()
    grad = ca.gradient(softmin_expr, dists)
    grad_time = (time.time() - t0) * 1e3

    t0 = time.time()
    hess = ca.hessian(softmin_expr, dists)[0]
    hess_time = (time.time() - t0) * 1e3

    # Create callable functions
    f_softmin = ca.Function("f_softmin", [dists], [softmin_expr])
    f_grad = ca.Function("f_grad", [dists], [grad])
    f_hess = ca.Function("f_hess", [dists], [hess])

    # Evaluate numerically
    sm_val = float(f_softmin(d_test))
    grad_val = np.array(f_grad(d_test)).flatten()
    grad_norm = np.linalg.norm(grad_val)

    print(f"{p:4d} | {grad_time:14.3f} | {hess_time:14.3f} | {grad_norm:10.3e} | {sm_val:10.6f}")
