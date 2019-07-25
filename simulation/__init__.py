from simulation.objects import Ray, Fiber
import math
import numpy as np
import matplotlib.pyplot as plt
import time

BEAD_SIZE = 2e-3


def generate_rays_multiple_sources(initial_points, bead_sizes, psis_cutoff, samples):
    rays = np.array([])
    for i in range(initial_points.shape[0]):
        rays = np.append(rays, generate_rays_single_source(initial_points[i], bead_sizes[i], psis_cutoff[i], samples))
    return rays


# Use fibonacci sphere algorithm optimize uniform distribution of 'samples' number of points on spherical cap
def generate_rays_single_source(initial_point, bead_size, psi_cutoff=math.pi, samples=100000):
    # TODO: parallelize
    rnd = 1.
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))

    # TODO: optimize this
    rays = np.array([Ray(initial_point) for i in range(samples)])

    for i in range(samples):
        z = - (((i * offset) - 1) + (offset / 2))
        r = math.sqrt(1 - pow(z, 2))
        psi = math.atan2(r, z)
        if psi > psi_cutoff:  # zenith angle
            rays = rays[:i]
            break
        theta = ((i + rnd) % samples) * increment  # azimuthal (projection) angle
        rays[i].theta = theta
        rays[i].psi = psi

    return rays


def probe_lengths(rays):
    # plot histogram of lengths of rays reflected within the fiber to see if there's a pattern
    pass


def create_image(end_points, savedir='./simulated_calibation/', draw=False):
    fig2 = plt.figure()
    # TODO: adjust bins
    heatmap, xedges, yedges = np.histogram2d(end_points[:, 0], end_points[:, 1], bins=150)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    i = 5
    plt.imsave(savedir + 'img_test_' + str(i) + '.tiff', heatmap.T)
    if draw:
        plt.show()


if __name__ == '__main__':
    start = time.time()

    fiber = Fiber()
    fig = plt.figure()
    fiber.draw(fig)

    # TODO: implement bead size by sampling initial positions from within size of bead
    # init_points = np.array([
    #     # [0e-6, 0e-6, 0e-6],
    #     [0, 50e-6, -50e-6],
    #     # [75e-6, 0, 0],
    #     # [10e-6, 0, 0]
    # ])

    # bead_sizes = np.repeat(BEAD_SIZE, init_points.shape[0])
    # # TODO: find a better way to store psi_max
    # # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html
    psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index)
    # psi_max = np.repeat(psi_max, init_points.shape[0])
    # for i, init_point in enumerate(init_points):
    #     if init_point[2] == 0:
    #         psi_max[i] = np.pi/2 - np.arcsin(fiber.cladding_index / fiber.core_index)  # pi/2 - theta_c
    #
    # nums = int(3e3)
    # generated_rays = generate_rays_multiple_sources(init_points, bead_sizes, psi_max, nums)

    sample_ray = Ray(np.array([0, 0, -300e-6]), 0, np.pi/4)
    refracted_ray = fiber.refract(sample_ray)
    sample_ray.draw(fig, refracted_ray.start)
    reflected_ray = fiber.reflect(refracted_ray)
    refracted_ray.draw(fig, reflected_ray.start)
    plt.show()

    fiber.propagate(refracted_ray, fig, draw=True)  # TODO: set psi limit for beads outside fiber
    # final_positions = fiber.propagate(generated_rays[0], fig, draw=True)
    # for i in range(1, generated_rays.size):
    #     final_positions = np.vstack((final_positions, fiber.propagate(generated_rays[i], fig, draw=True)))
    #     if ((i - 1) * 100) % generated_rays.size > (i * 100) % generated_rays.size:
    #         print('progress: ', int((i / generated_rays.size) / 0.01), '%')
    plt.show()

    # savedir = '/Users/Admin/Google Drive/Year4/Y4S3/ESC499 - Thesis/code/simulation/simulated_calibration/'
    # create_image(end_points=final_positions, savedir=savedir, draw=True)

    end = time.time()
    print("Simulation time: ", end-start)
