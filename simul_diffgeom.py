import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
from sympy.diffgeom import *
from matplotlib import pyplot as plt

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

# Transform time coordinate to write optical metric more concisely
dtprime = 1/n * dt - dr * ((1 - f)*(n**2 - f)/(c*f))**.5 / (c*f)
print('dtprime:', dtprime, '\n')

# Define the optical metric g
g = f*c**2 * TensorProduct(dtprime, dtprime) - 1/f * TensorProduct(dr, dr) \
    - r**2 * TensorProduct(dtheta, dtheta) \
    - r**2*sin(theta)**2 * TensorProduct(dphi, dphi)

print('optical metric g:', twoform_to_matrix(g), '\n')

# Calculate the Christoffel symbols
christoffels = metric_to_Christoffel_2nd(g)
print('christoffel symbols:', christoffels)

# Compute ricci tensor
# ricci = metric_to_Ricci_components(g)
# print(ricci)

christ_lamb = lambdify((c, G, M, n, r, theta), christoffels, modules='numpy')

def F(t, y):
    u = np.array(y[0:4])
    v = np.array(y[4:8])

    subs = christ_lamb(1, 1, 1, 1.2, u[1], u[2])

    du = v
    dv = -np.dot(np.dot(subs, v), v) # Contraction of christoffel symbols with velocity vectors

    return np.concatenate((du, dv))

T = 100
t_eval = np.linspace(0, T, 1000)
initial_value = [0, 10, np.pi/2, 0, 1, 0, 0, 0]

soln = solve_ivp(F, [0, T], initial_value, t_eval=t_eval)

plt.plot(soln.y[0], soln.y[1])
plt.ylabel('r')
plt.xlabel('t')
plt.show()
