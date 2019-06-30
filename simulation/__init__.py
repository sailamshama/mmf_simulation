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

    # psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index) # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html
    # ray_defining_points = fibonacci_sphere(samples = ray_density, randomize = False, psi_max)

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

if __name__ == '__main__':

    initial_points = np.array([[0e-6, 0e-6, -0.000000001e-6], [50e-6, 0e-6, -0.000000001e-6], [99e-6, 0e-6, -0.000000001e-6]])
    rays = generate_rays_multiple_sources(initial_points, 100, math.pi / 2)
    print(rays[1][20].start, rays[2][20].start, rays[0][20].start, rays[0][20].theta, rays[0][20].psi)