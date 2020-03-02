import numpy as np
from matplotlib import pyplot as plt


####################
### Define stuff ###
####################

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


### Constants ###
mu_0 = 1

### Dipole ###
dipole_magnitude = 1
tilt = 20
tilt = np.radians(tilt) # angle between z-axis, tiltend about y-axis
dp_x = np.sin(tilt)*dipole_magnitude
dp_y = 0
dp_z = np.cos(tilt)*dipole_magnitude
dipole = np.array([dp_x, dp_y, dp_z])

### Particles ###
particle_mass = 1
particle_charge = 1

### Space ###
xyz_lim = 20
N = 100
x = np.linspace(-xyz_lim, xyz_lim, N)
y = np.linspace(-xyz_lim, xyz_lim, N)
z = np.linspace(-xyz_lim, xyz_lim, N)


######################
### Simulate stuff ###
######################

##################
### Plot stuff ###
##################

### B-field ###
xx, yy = np.meshgrid(x, y)
B_xx, B_yy, B_zz = B_field(xx, yy, np.zeros_like(xx))
plt.streamplot(xx, yy, B_xx, B_yy)
plt.show()
