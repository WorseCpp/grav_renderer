import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.patches as patches

import matplotlib.pyplot as plt


dd = lambda u : 1.5 * u**2 - u

u0 = 1.1
phi0 = 0.0

r0 = 1 / u0
x0 = r0 * np.cos(phi0)
y0 = r0 * np.sin(phi0)

v = [.99, 0.01]  # Initial velocity in polar coordinates (dr/dphi, d^2r/dphi^2)
h = .01
x1 = x0 + h * v[0]
y1 = y0 + h * v[1]

r1 = np.sqrt(x1**2 + y1**2)
u1 = 1 / r1  # Convert to u = 1/r
phi1 = np.arctan2(y1, x1)

# Velocity in polar coordinates
du_by_dphi = (u1 - u0) / (phi1 - phi0)

def odefun(phi, y):
    # y[0] = r, y[1] = dr/dphi
    return [y[1], dd(y[0])]

phi_span = (phi0, phi0 + .01)  # integrate from phi0 to 10
y0 = [u0, du_by_dphi]

sol = solve_ivp(odefun, phi_span, y0, method='RK45', dense_output=True)

phi_vals = np.linspace(phi_span[0], phi_span[1], 300)
u_vals = sol.sol(phi_vals)[0]
r_vals = 1 / u_vals  # Convert to r = 1/u
x_vals = r_vals * np.cos(phi_vals)
y_vals = r_vals * np.sin(phi_vals)

plt.figure()

plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Plot of r,phi Pairs')
ax = plt.gca()
max_r = 10
# Draw angular grid lines every 30°
for angle in np.deg2rad(np.arange(0, 360, 30)):
    ax.plot([0, np.cos(angle) * max_r], [0, np.sin(angle) * max_r],
            color='gray', linestyle='--', linewidth=0.5)
# Draw radial grid circles (avoid drawing a circle at r=0)
for r_circle in np.linspace(0, max_r, 5)[1:]:
    circle = patches.Circle((0, 0), r_circle, fill=False,
                            edgecolor='gray', linestyle='--', linewidth=0.5)
    ax.add_patch(circle)

plt.plot(x_vals, y_vals)
plt.axis('equal')
plt.xlim(-max_r, max_r)
plt.ylim(-max_r, max_r)



plt.show()

