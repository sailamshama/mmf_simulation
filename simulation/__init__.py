from simulation.ray import Ray
import vectormath as vmath
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DEBUG=1

def generate_rays(initial_points, samples, psi_max=math.pi):

    # Use fibonacci sphere algorithm optimize uniform distribution of 'samples' number of points on spherical cap.
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    rnd = 1.
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))
    points = np.array([])

    for i in range(samples):
        z = - (((i * offset) - 1) + (offset / 2))
        r = math.sqrt(1 - pow(z, 2))
        psi = math.atan2(r, z)
        if psi > psi_max:
           break
        theta = ((i + rnd) % samples) * increment
        x = math.cos(theta) * r
        y = math.sin(theta) * r

        if points.size == 0:
            points = np.array([x,y,z])
        else:
            points = np.vstack((points, [x, y, z]))

    # psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index) # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html
    # ray_defining_points = fibonacci_sphere(samples = ray_density, randomize = False, psi_max)

    pass


if __name__ == '__main__':
    initial_points = vmath.Vector3Array()

    generate_rays(initial_points, samples = 10000)

    # r0 = -z0 * np.tan(theta_max)
    # r = np.linspace(r0, 0, mesh_density, endpoint=False)
    # r_space = r[0] - r[1]
    # mesh = np.array([[0, 0, 0]])
    # origin = init_pos[:2]

    points = fibonacci_sphere(samples=10000 , psi_max = math.pi / 4)
    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.scatter3D(points[:,0], points[:,1], points[:,2], zdir='z', s=1, c=None)
    fig.show()
