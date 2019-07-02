from simulation.ray import Ray
import vectormath as vmath
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DEBUG = 1

def generate_rays_multiple_sources(initial_points, samples, psi_max=math.pi):

    rays_multiple_points = np.array([])
    for initial_point in initial_points:
        rays_single_point = generate_rays_single_source(initial_point, psi_max, samples)
        if np.size(rays_multiple_points) == 0:
            rays_multiple_points = rays_single_point
        else:
            rays_multiple_points = np.vstack((rays_multiple_points, rays_single_point))

    return rays_multiple_points

# Use fibonacci sphere algorithm optimize uniform distribution of 'samples' number of points on spherical cap

def generate_rays_single_source(initial_point, psi_max = math.pi, samples = 1000):
    rnd = 1.
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))
    rays = np.array([])

    for i in range(samples):
        z = - (((i * offset) - 1) + (offset / 2))
        r = math.sqrt(1 - pow(z, 2))
        psi = math.atan2(r, z)
        if psi > psi_max:
            break
        theta = ((i + rnd) % samples) * increment
        if rays.size == 0:
            rays = np.array([Ray(initial_point, theta, psi)])
        else:
            rays = np.append(rays, np.array([Ray(initial_point, theta, psi)]))

    return rays

def get_intersection(ray, fiber):
    pass

if __name__ == '__main__':

    initial_points = np.array([[0e-6, 0e-6, -0.000000001e-6], [50e-6, 0e-6, -0.000000001e-6], [99e-6, 0e-6, -0.000000001e-6]])
    # psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index) # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html
    rays = generate_rays_multiple_sources(initial_points, 100, math.pi / 2)


    # draw cylinder
    # source: https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.linspace(-1, 1, 100)
    z = np.linspace(-2, 2, 100)
    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt(1 - Xc ** 2)
    rstride = 20
    cstride = 10
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
    ax.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
    t = 1
    # draw rays




