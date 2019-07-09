from simulation.objects import Ray, Fiber
import math
import numpy as np
import matplotlib.pyplot as plt

def generate_rays_multiple_sources(initial_points, samples, psi_cutoff=math.pi):
    rays_multiple_points = np.array([])
    for initial_point in initial_points:
        rays_single_point = generate_rays_single_source(initial_point, psi_cutoff, samples)
        if np.size(rays_multiple_points) == 0:
            rays_multiple_points = rays_single_point
        else:
            rays_multiple_points = np.vstack((rays_multiple_points, rays_single_point))
    return rays_multiple_points


# Use fibonacci sphere algorithm optimize uniform distribution of 'samples' number of points on spherical cap
def generate_rays_single_source(initial_point, psi_cutoff=math.pi, samples=1000):
    rnd = 1.
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))

    rays = np.array([Ray(initial_point) for i in range(samples)])

    for i in range(samples):
        z = - (((i * offset) - 1) + (offset / 2))
        r = math.sqrt(1 - pow(z, 2))
        psi = math.atan2(r, z)
        if psi > psi_cutoff: #zenith angle
            rays = rays[:i]
            break
        theta = ((i + rnd) % samples) * increment #azimuthal (projection) angle

        rays[i].theta = theta
        rays[i].psi = psi

    return rays

def probe_lengths(rays):
    # plot histogram of lengths of rays reflected within the fiber to see if there's a pattern
    pass

def create_image(end_points):
    pass

if __name__ == '__main__':

    fiber = Fiber()
    init_points = np.array([[-50e-6, 0e-6, 0e-6]])
    # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html
    # psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index)
    psi_max = np.pi/ 2
    # generated_rays = generate_rays_multiple_sources(init_points, 100000, psi_max)
    generated_rays = generate_rays_single_source(np.squeeze(init_points), psi_max, 1000)

    # TODO: debug this
    points = np.zeros((generated_rays.size, 3))
    for i, ray in enumerate(generated_rays):
        points[i] = np.array([np.sin(ray.psi) * np.cos(ray.theta), np.sin(ray.psi) * np.sin(ray.theta), np.cos(ray.psi)])
        print(points[i])

    generated_rays_figure = plt.figure()
    ax = plt.gca(projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.scatter3D(points[:,0], points[:,1], points[:, 2], zdir='z', s=1, c=None)
    plt.show()
    plt.close(generated_rays_figure)

    end_points = fiber.propagate(generated_rays)


