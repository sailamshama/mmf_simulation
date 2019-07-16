from simulation.objects import Ray, Fiber
import math
import numpy as np
import matplotlib.pyplot as plt


def generate_rays_multiple_sources(initial_points, samples, psi_cutoff=math.pi):
    # TODO: test and debug this
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
    heatmap, xedges, yedges = np.histogram2d(final_positions[:, 0], final_positions[:, 1], bins=150)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    i = 1
    plt.imsave(savedir + 'img_test_' + str(i) + '.tiff', heatmap.T)
    if draw:
        plt.show()


if __name__ == '__main__':

    fiber = Fiber()
    fig = plt.figure()
    # fiber.draw(fig)

    # TODO: implement bead size by sampling initial positions from within size of bead
    init_points = np.array([[0e-6, 0e-6, 0e-6], [50e-6, 0, 0e-6], [75e-6, 0, 0], [10e-6, 0, 0]])
    # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html

    # TODO: find a better way to store psi_max
    psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index)
    psi_max = np.repeat(psi_max, init_points.shape[0])
    for i, init_point in enumerate(init_points):
        if init_point[2] == 0:
            psi_max[i] = np.pi/2 - np.arcsin(fiber.cladding_index / fiber.core_index)  # pi/2 - theta_c

    generated_rays = generate_rays_single_source(init_points[3], psi_max[3], 10000000)

    final_positions = fiber.propagate(generated_rays[i], fig, draw=False)
    for i in range(0, generated_rays.size):
        final_positions = np.vstack((final_positions, fiber.propagate(generated_rays[i], fig, draw=False)))
    plt.show()

    savedir = '/Users/Admin/Google Drive/Year4/Y4S3/ESC499 - Thesis/code/simulation/simulated_calibration/'
    create_image(end_points=final_positions, savedir=savedir, draw=True)
