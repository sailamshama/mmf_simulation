import math, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

def fibonacci_sphere(samples=1, randomize=True, psi_threshold = math.pi):

    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        z = - (((i * offset) - 1) + (offset / 2))

        r = math.sqrt(1 - pow(z, 2)) # assuming unit sphere
        psi = math.atan2(r, z) # zenith angle
        # psi = np.tan(r/z)
        if psi > psi_threshold:  # zenith angle
            break
        theta = ((i + rnd) % samples) * increment # azimuthal (projection) angle

        x = math.cos(theta) * r
        y = math.sin(theta) * r

        points.append([x, y, z])

    rnd = 1.
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))

    return points

if __name__ == '__main__':

    points = np.array(fibonacci_sphere(10000, False, math.pi/2))
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], zdir='z', s=1, c=None)
    plt.show()
    plt.close()
