# Black Hole hydrodynamical simulation
# Written by Jack Bishop in 2023 for AMATH361

import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
from sympy.diffgeom import *
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

print('Hydrodynamic Schwarzchild Black Hole Optical Simulation')
print('-------------------------------------------------------', '\n')

# Define manifold and coordinate system
m = Manifold('Optical Schwarzchild', 4)
p = Patch('origin', m)
cs = CoordSystem('spherical', p,
                 [Symbol('t', real=True), Symbol('r', real=True),
                  Symbol('theta', real=True), Symbol('phi', real=True)])

t, r, theta, phi = cs.coord_functions()
dt, dr, dtheta, dphi = cs.base_oneforms()

# Some symbols related to the Schwarzchild black hole
r0, G, M, c, n = symbols('r0 G M c n')
r0 = 2*G*M/c**2
f = 1 - r0/r
print('f:', f, '\n')

# Fluid velocity
beta = ((1 - f)/(n**2 - f))**.5

# Transform time coordinate to write optical metric more concisely
dtprime = 1/n * dt - dr * ((1 - f)*(n**2 - f)/(c*f))**.5 / (c*f)
# print('dtprime:', dtprime, '\n')

# Define the optical metric g
g = f*c**2 * TensorProduct(dtprime, dtprime) - 1/f * TensorProduct(dr, dr) \
    - r**2 * TensorProduct(dtheta, dtheta) \
    - r**2*sin(theta)**2 * TensorProduct(dphi, dphi)

# print('optical metric g:', twoform_to_matrix(g), '\n')

# Calculate the Christoffel symbols
christoffels = metric_to_Christoffel_2nd(g)
# print('christoffel symbols:', christoffels)

# Compute ricci tensor
# ricci = metric_to_Ricci_components(g)
# print(ricci)

christ_lamb = lambdify((c, G, M, n, r, theta), christoffels, modules='numpy')
r0_lamb = lambdify((c, G, M), r0, modules='numpy')
beta_lamb = lambdify((c, G, M, n, r), beta, modules='numpy')

# TODO: come up with more realistic values for these constants
subs_c = 1.
subs_G = 0.1
subs_M = 1.
subs_n = 1.2

def F(t, x):
    coords = np.array(x[0:4])
    fourvel = np.array(x[4:8])
    subs = christ_lamb(subs_c, subs_G, subs_M, subs_n, coords[1], coords[2])
    a = -np.dot(np.dot(subs, fourvel), fourvel) # Contraction of christoffel symbols with velocity vectors
    # This gives us the acceleration in the geodesic equation

    return np.concatenate((fourvel, a))

# Compute schwarzchild radius
subs_r0 = r0_lamb(subs_c, subs_G, subs_M)
# print('Schwarzchild rad:', subs_r0)

T = 1000
t_eval = np.linspace(0, T, 10000)
phidot_lamb = lambdify((c, G, M, n), 2/(3*r0*n) * (1/3)**.5) # This is the azimuthal velocity which will normalize u under the optical metric
subs_phidot = phidot_lamb(subs_c, subs_G, subs_M, subs_n)
initial = [0, subs_r0 * 3 / 2, np.pi/2, 0, 1, 0, 0, subs_phidot] # Initial posn, velocity (spherical coords)

soln = solve_ivp(F, [0, T], initial, t_eval=t_eval)
# print('soln:', soln.y)

soln_t = soln.y[0]
soln_r = soln.y[1]
soln_theta = soln.y[2]
soln_phi = soln.y[3]
# print('r:', soln_r)
# print('theta:', soln_theta)
# print('phi:', soln_phi)

# Convert spherical coordinates to cartesian
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return (x, y, z)

soln_cart = spherical_to_cartesian(soln_r, soln_theta, soln_phi)
soln_x = soln_cart[0]
soln_y = soln_cart[1]
soln_z = soln_cart[2]

# Mesh representing event horizon at Schwarzchild radius
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j] # u, v map
horiz_wireframe = spherical_to_cartesian(subs_r0, v, u)

# Compute velocity field for fluid
grid_x, grid_y, grid_z = np.mgrid[-.5:.5:10j, -.5:.5:10j, -.5:.5:10j]
grid_x = grid_x.reshape([1, 1000])[0]
grid_y = grid_y.reshape([1, 1000])[0]
grid_z = grid_z.reshape([1, 1000])[0]
grid_norm = (grid_x**2 + grid_y**2 + grid_z**2)**.5
fluid_vel = np.array([beta_lamb(subs_c, subs_G, subs_M, subs_n, grid_r) for grid_r in grid_norm])
fluid_vel_x = grid_x * fluid_vel / grid_norm
fluid_vel_y = grid_y * fluid_vel / grid_norm
fluid_vel_z = grid_z * fluid_vel / grid_norm

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(horiz_wireframe[0], horiz_wireframe[1], horiz_wireframe[2], color='black')
ax.quiver(grid_x, grid_y, grid_z, fluid_vel_x, fluid_vel_y, fluid_vel_z, length=0.1, colors=plt.cm.hsv(fluid_vel/1.1))
ax.plot(soln_x, soln_y, soln_z, color='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
ax.set_zlim(-.5, .5)
ax.set_aspect('equal')
plt.show()
