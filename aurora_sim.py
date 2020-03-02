import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import mu_0, elementary_charge, proton_mass


####################
### Define stuff ###
####################


### Dipole ###
dipole_magnitude = 1
tilt = 20 # degrees, arb. chosen
tilt = np.radians(tilt) # angle between z-axis, tiltend about y-axis
dp_x = np.sin(tilt)*dipole_magnitude
dp_y = 0
dp_z = np.cos(tilt)*dipole_magnitude
dipole = np.array([dp_x, dp_y, dp_z])

### Space ###
xyz_lim = 20
N = 100
x = np.linspace(-xyz_lim, xyz_lim, N)
y = np.linspace(-xyz_lim, xyz_lim, N)
z = np.linspace(-xyz_lim, xyz_lim, N)

### simulation params ###
dt = 0.1
SIM_TIME = 1000
INITIAL = [-10, 0, 0, 2, 0, 0] # [xo, y0, z0, dx0/dt, dy0/dt, dz0/dt]


### Functions ###
def B_field(x, y, z):
    """ Computes B-field from a dipole in position (x,y,x) """
    r_vec = np.array([x, y, z])
    r = np.maximum(np.sqrt(x**2 + y**2 + z**2), 0.001) # hackaround to avoid mess inn origo :p
    #dip_dot_r = np.dot(r_vec, dipole)
    dip_dot_r = x*dp_x + y*dp_y + z*dp_z
    B_x = (mu_0/(4*np.pi)) * ( ( (3*x*dip_dot_r)/(r**5) ) - (dp_x)/(r**3) )
    B_y = (mu_0/(4*np.pi)) * ( ( (3*y*dip_dot_r)/(r**5) ) - (dp_y)/(r**3) )
    B_z = (mu_0/(4*np.pi)) * ( ( (3*z*dip_dot_r)/(r**5) ) - (dp_z)/(r**3) )
    B = np.array([B_x, B_y, B_z])
    return B


def ddt_vel_accl(t, y):
    """ equation of motion to be solved
        Parameters:
            t : time, scalar
            y : position/velocity, vector, [x, y, z, dx/dt, dy/dy, dz/dt]
        Returns:
            [velocity, acceleration] : velocity and acceleration in a list at current time t
    """
    velocity = np.array(y[3:])
    B = B_field(y[0], y[1], y[2])
    acceleration = elementary_charge*np.cross(velocity, B)/proton_mass
    return np.concatenate((velocity, acceleration))
    #return [velocity, acceleration]


# damn, how does this work in 3D?? :000
def solve_equation_of_motion(ddt_vel_accl, SIM_TIME, INITIAL, dt):
    sol = solve_ivp(ddt_vel_accl, [0, SIM_TIME], INITIAL, max_step=dt)
    path, velocities = sol.y[0], sol.y[1]
    return path, velocities


######################
### Simulate stuff ###
######################
p, v = solve_equation_of_motion(ddt_vel_accl, SIM_TIME, INITIAL, dt)
print(p)
print(p.shape[0])
print(SIM_TIME/dt)


##################
### Plot stuff ###
##################

### B-field ###
xx, yy = np.meshgrid(x, y)
B_xx, B_yy, B_zz = B_field(xx, yy, np.zeros_like(xx))
plt.streamplot(xx, yy, B_xx, B_yy)
#plt.plot(x, p.T[1])
plt.show()



