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
    rays = np.array([])

    for i in range(samples):
        z = - (((i * offset) - 1) + (offset / 2))
        r = math.sqrt(1 - pow(z, 2))
        psi = math.atan2(r, z)
        if psi > psi_cutoff:
            break
        theta = ((i + rnd) % samples) * increment
        if rays.size == 0:
            rays = np.array([Ray(initial_point, theta, psi)])
        else:
            rays = np.append(rays, np.array([Ray(initial_point, theta, psi)]))

    return rays


if __name__ == '__main__':

    fiber = Fiber()

    init_points = np.array([[-50e-6, 0e-6, 0e-6]])
    # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html
    psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index)
    generated_rays = generate_rays_multiple_sources(init_points, 10000, psi_max)

    fig = plt.figure()
    fiber.draw(fig)
    ray = Ray(np.array([0.000000,0.000000,0.000000]), 0, np.pi/ 2) #should return (ellipse.a, 0, 0)
    final_point = fiber.get_intersection(ray) # should get [ellipse.a, 0,0]
    # final_point = np.array([1,1,1])
    ray.draw(fig, final_point)
    plt.show()

    # end_points = np.array([])
    # for ray in generated_rays:
    #     while ray.start[3] < fiber.length: # TODO: watch out of infinite loop
    #         reflected_ray = ray.reflected(fiber) # TODO: when reflected_ray.start > fiber.length
    #         ray.draw(reflected_ray.start)
    #         ray = reflected_ray
    #     end_points.append(ray.start)
    # TODO: draw histogram of end_points

    t = 1


