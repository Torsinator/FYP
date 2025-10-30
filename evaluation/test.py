import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Simulate a few sample "distances" across horizon steps
# (think: squared distance to target at each predicted timestep)
# ------------------------------------------------------------
x = np.linspace(0, 10, 100)       # horizon step index (or just a distance variable)
d1 = (x - 2.0)**2 + 1.0           # first trajectory (minimum at x=2)
d2 = (x - 7.0)**2 + 0.2           # second trajectory (minimum at x=7)

# Combine into vector of distances at each k
dists = np.vstack([d1, d2])       # shape (2, 100)
true_min = np.min(dists, axis=0)  # the true elementwise minimum

# ------------------------------------------------------------
# Soft-min approximation
# ------------------------------------------------------------
def softmin_pnorm(d, p, eps=1e-6):
    """p-norm approximation to min over axis 0"""
    return (np.sum((d + eps)**(-p), axis=0))**(-1/p)

# Try several p values
p_values = [2, 5, 10, 20, 50]
colors = plt.cm.viridis(np.linspace(0, 1, len(p_values)))

plt.figure(figsize=(8, 5))
plt.plot(x, true_min, 'k--', lw=2, label="True min")

for p, c in zip(p_values, colors):
    sm = softmin_pnorm(dists, p)
    plt.plot(x, sm, color=c, lw=2, label=f"p = {p}")

plt.xlabel("x (index or distance variable)")
plt.ylabel("Soft-min value")
plt.title("Soft-min (p-norm) approximation to min")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
