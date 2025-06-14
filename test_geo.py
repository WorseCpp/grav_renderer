import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.patches as patches

import matplotlib.pyplot as plt

M = 1e-10

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


hist = np.array(
[[1.79099, 0.1],
[3.46409, 0.2],
[5.10257, 0.3],
[6.69007, 0.4],
[8.21073, 0.5],
[9.64935, 0.6],
[10.9916, 0.7],
[12.2239, 0.8],
[13.3342, 0.9],
[14.3112, 1],
[15.1452, 1.1],
[15.8279, 1.2],
[16.3525, 1.3],
[16.7136, 1.4],
[16.9078, 1.5],
[16.933, 1.6],
[16.789, 1.7],
[16.4773, 1.8],
[16.001, 1.9],
[15.3647, 2],
[14.575, 2.1],
[13.6396, 2.2],
[12.568, 2.3],
[11.3707, 2.4],
[10.0599, 2.5],
[8.64853, 2.6],
[7.15076, 2.7],
[5.58154, 2.8],
[3.95654, 2.9],
[2.29202, 3],
[0.604598, 3.1],
[0.265847, 3.12],
[0.19808, 3.124],
[0.130309, 3.128],
[0.0625366, 3.132],
[0.0489819, 3.1328],
[0.0354272, 3.1336],
[0.0218725, 3.1344],
[0.00831772, 3.1352],
[0.00560677, 3.13536],
[0.00289582, 3.13552],
[0.000184869, 3.13568]]
)


sol = solve_ivp(odefun, phi_span, c_y0, method='BDF', dense_output=True)
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

u_hist = hist[:,0]
phi_hist = hist[:, 1]

plt.scatter((1/u_hist) * np.cos(phi_hist), (1/u_hist) * np.sin(phi_hist), s = 10)

plt.plot(x0, y0, 'ro', label='Starting Point')
plt.arrow(x0, y0, 10 * (x1 - x0), (y1 - y0) * 10, head_width=0.5, head_length=1., fc='blue', ec='blue', label='Initial Velocity')


plt.plot(x_vals, y_vals)
plt.axis('equal')
plt.xlim(-max_r, max_r)
plt.ylim(-max_r, max_r)



plt.show()

