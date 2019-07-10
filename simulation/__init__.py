from simulation.objects import Ray, Fiber
import math
import numpy as np
import matplotlib.pyplot as plt

def generate_rays_multiple_sources(initial_points, samples, psi_cutoff=math.pi):
    #TODO: fix this too
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

    rays = np.array([Ray(initial_point) for i in range(samples)]) # TODO: optimize this

    for i in range(samples):
        z = - (((i * offset) - 1) + (offset / 2))
        r = math.sqrt(1 - pow(z, 2))
        psi = math.atan2(r, z)
        if psi > psi_cutoff: # zenith angle
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
    fig = plt.figure()
    fiber.draw(fig)

    # init_points = np.array([[0e-6, 0e-6, 0e-6], [0, 0, -50e-6]])
    # # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html
    #
    # psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index)
    # psi_max = np.repeat(psi_max, init_points.shape[0])
    # for i, init_point in enumerate(init_points):
    #     if init_point[2] == 0:
    #         psi_max[i] = np.pi/2 - np.arcsin(fiber.cladding_index / fiber.core_index) # pi/2 - theta_c
    #
    # generated_rays = generate_rays_single_source(init_points[0], psi_max[0], 1000)

    # TODO: draw both ray propagations
    # TODO: fix overwriting ray objects
    ray1 = Ray(np.array([0, 0, 0]), 0, 0)
    ray2 = Ray(np.array([0, 0, 0]), np.pi/4, np.pi/20)
    # TODO: fix this too
    # end_point1 = fiber.propagate(ray1, fig, draw=True)
    end_point2 = fiber.propagate(ray2, fig, draw=True)
    # fiber.propagate(generated_rays[2], fig, draw=True)

    # for i in range(1, generated_rays.size):
    #     final_points = np.append(final_points, fiber.propagate(generated_rays[i], fig, draw=True))

    plt.show()
    #
    # fig2 = plt.figure()
    # xs = final_points[:0]
    # ys = final_points
    # plt.hist2d(xs, ys)
    # plt.show()

    # end_points = np.array([])
    # for ray in generated_rays: # TODO: watch out of infinite loop
    #     while ray.start[2] < fiber.length:
    #         reflected_ray = ray.reflected(fiber) # TODO: when reflected_ray.start > fiber.length
    #         ray.draw(reflected_ray.start)
    #         ray = reflected_ray
    #     end_points.append(ray.start)
    t = 1

