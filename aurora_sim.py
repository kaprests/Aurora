import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import mu_0, elementary_charge, proton_mass

####################
### Define stuff ###
####################

### Physical parameters ###
''' Dipole '''
dipole_magnitude = 1e5 # No physical reasoning behind this number
tilt = 11 # degrees
tilt = np.radians(tilt) # angle between z-axis, tiltend about y-axis
dp_x = np.sin(tilt)*dipole_magnitude
dp_y = 0
dp_z = np.cos(tilt)*dipole_magnitude
dipole = np.array([dp_x, dp_y, dp_z])


R = 6.37e6 # meter, earth's radius
R_sim = 10 # Units of space, earth's radius in simulation
sim_space_unit = R/R_sim # Number of meters per space unit
V0 = 450*1000 # Thypical solar wind speeds ~ 10km/h, earth radii per second (m/s)
V0_sim = V0/sim_space_unit # Units of space per second
print("Units of space per second: ", V0_sim)

### Space ###
N = 100
x = np.linspace(-80, 20, N)
y = np.linspace(-80, 20, N)
z = np.linspace(-40, 60, N)

### simulation params ###
dt = 1
SIM_TIME = 500 # dts
INITIALS = [
        [-20, -20, 40, V0_sim, V0_sim, 0],
        [-20, -20, 30, V0_sim, V0_sim, 0],
        [-20, -20, 20, V0_sim, V0_sim, 0],
        [-20, -20, 10, V0_sim, V0_sim, 0],
        [-20, -20, -10, V0_sim, V0_sim, 0],
        ] # [xo, y0, z0, dx0/dt, dy0/dt, dz0/dt]


### Functions ###
def B_field(x, y, z):
    """ Computes B-field from a dipole in position (x,y,x) """
    r = np.maximum(np.sqrt(x**2 + y**2 + z**2), 0.001) # hackaround to avoid mess inn origo :p
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
            [vx, vy, vz, ax, ay, az] : velocity and acceleration in a list at current time t
    """
    velocity = np.array(y[3:])
    B = B_field(y[0], y[1], y[2])
    acceleration = elementary_charge*np.cross(velocity, B)/proton_mass
    return np.concatenate((velocity, acceleration))


def solve_equation_of_motion(ddt_vel_accl, SIM_TIME, INITIAL, dt):
    sol = solve_ivp(ddt_vel_accl, [0, SIM_TIME], INITIAL, max_step=dt)
    path, velocities = sol.y[:3], sol.y[3:]
    return path, velocities


######################
### Simulate stuff ###
######################

paths = []
velocities = []

for initial in INITIALS:
    path, velocity = solve_equation_of_motion(ddt_vel_accl, SIM_TIME, initial, dt)
    paths.append(path)
    velocities.append(velocity)

#print("Velocity: ", V0)
#print("Simulation time: ", SIM_TIME)
#print("distance units moved (if straight line): ", np.sqrt((path[0][-1]-path[0][0])**2 + (path[1][-1]-path[1][0])**2 + (path[2][-1]-path[2][0])**2))


##########################
### Save paths to file ###
##########################

for i in range(len(paths)):
    z0 = INITIALS[i][2]
    np.savetxt(str(z0) + "path.csv", paths[i], delimiter=',')

##################
### Plot stuff ###
##################

### B-field xz-plane ###
xx, zz = np.meshgrid(x, z)
B_xx, B_yy, B_zz = B_field(xx, np.zeros_like(xx), zz)
plt.streamplot(xx, zz, B_xx, B_zz)
earth = plt.Circle((0, 0), R_sim, color="blue")
plt.gca().add_artist(earth)

for i in range(len(paths)):
    path = paths[i]
    x0 = (INITIALS[i][0], INITIALS[i][1], INITIALS[i][2])
    v0 = (str(INITIALS[i][3])[:4], str(INITIALS[i][4])[:4], str(INITIALS[i][5])[:4])
    plt.plot(path[0], path[2], label="Start pos: " + str(x0) + "v0: " + str(v0))
plt.legend()
plt.xlabel("x [Earth radii]")
plt.ylabel("z [Earth radii]")
plt.savefig("xz-plane.pdf")
plt.show()


yy, zz = np.meshgrid(y, z)
B_xx, B_yy, B_zz = B_field(np.zeros_like(yy), yy, zz)
plt.streamplot(yy, zz, B_yy, B_zz)
earth = plt.Circle((0, 0), R_sim, color="blue")
plt.gca().add_artist(earth)

for i in range(len(paths)):
    path = paths[i]
    x0 = (INITIALS[i][0], INITIALS[i][1], INITIALS[i][2])
    v0 = (str(INITIALS[i][3])[:4], str(INITIALS[i][4])[:4], str(INITIALS[i][5])[:4])
    plt.plot(path[1], path[2], label="Start pos: " + str(x0) + "v0: " + str(v0))
plt.legend()
plt.xlabel("y [Earth radii]")
plt.ylabel("z [Earth radii]")
plt.savefig("yz-plane.pdf")
plt.show()


# Plot the magnitude of the velocity of one of the particles
velocity = np.sqrt(velocities[0][0]**2+velocities[0][1]**2+velocities[0][2]**2)
time_vec = np.linspace(0, SIM_TIME, len(velocity))
plt.plot(time_vec, velocity)
plt.xlabel("time")
plt.ylabel("velocity")
plt.savefig("velocity_plot.pdf")
plt.show()


