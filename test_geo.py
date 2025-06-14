import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.patches as patches

import matplotlib.pyplot as plt

M = 1e-7

def u_phi_to_x_y(u, phi):
    r = 1 / u
    x = r * np.cos(phi)
    y = r*np.sin(phi)
    return x,y

def x_y_to_u_phi(x, y):
    r = (x**2 + y**2)**.5
    phi = np.atan2(y,x)
    return 1/r, phi

dd = lambda u : 3 * M * u**2 - u

x0, y0, x1, y1 = 10, 0,9.9999 ,5.89246e-07

xf, yf, xf1, yf1 = 8.03612, 4.28637, 8.03613, 4.28741

u0, phi0 = x_y_to_u_phi(x0,y0)

u1, phi1 = x_y_to_u_phi(x1, y1)

# Velocity in polar coordinates
du_by_dphi = (u1 - u0) / (phi1 - phi0)

def odefun(phi, y):
    # y[0] = r, y[1] = dr/dphi
    return [y[1], dd(y[0])]

phi_span = (phi0, phi0 + np.pi*1.5)  # integrate from phi0 to 10
c_y0 = [u0, du_by_dphi]





def event_stop(phi, y):
    return y[0] - 0.001

event_stop.terminal = True
event_stop.direction = -1
sol = solve_ivp(odefun, phi_span, c_y0,
                method='BDF',
                dense_output=True,
                events=event_stop,
                rtol=1e-8,
                atol=1e-10)

print(sol.t)
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
max_r = 10000 * .1
# Draw angular grid lines every 30°
for angle in np.deg2rad(np.arange(0, 360, 30)):
    ax.plot([0, np.cos(angle) * max_r], [0, np.sin(angle) * max_r],
            color='gray', linestyle='--', linewidth=0.5)
# Draw radial grid circles (avoid drawing a circle at r=0)
for r_circle in np.linspace(0, max_r, 5)[1:]:
    circle = patches.Circle((0, 0), r_circle, fill=False,
                            edgecolor='gray', linestyle='--', linewidth=0.5)
    ax.add_patch(circle)

circle = patches.Circle((0, 0), 2*M, fill=True,
                          edgecolor='black', facecolor='black',
                          linestyle='-', linewidth=1)
ax.add_patch(circle)


plt.plot(x0, y0, 'ro', label='Starting Point')
plt.arrow(x0, y0, 10 * (x1 - x0), (y1 - y0) * 10, head_width=0.5, head_length=1., fc='blue', ec='blue', label='Initial Velocity')


plt.plot(x_vals, y_vals)
plt.axis('equal')
plt.xlim(-max_r, max_r)
plt.ylim(-max_r, max_r)



plt.show()

