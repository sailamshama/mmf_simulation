from simulation.fiber import Ray, Fiber
from os import getcwd
import numpy as np
import matplotlib.pyplot as plt
import time
from simulation.bead import Bead
from mpl_toolkits.mplot3d import Axes3D

DRAW = False
FILENAME = 'mid'
BIN_SIZE = 150

if __name__ == '__main__':
    fig1 = plt.figure()
    fig2 = plt.figure()

    start = time.time()

    fiber = Fiber()
    if DRAW:
        fiber.draw(fig1)

    beads = [
        # Bead(np.array([0e-6, 0e-6, 0e-6])),
        # Bead(np.array([0e-6, 10e-6, 0e-6])),
        # Bead(np.array([0e-6, 20e-6, 0e-6])),
        # Bead(np.array([0e-6, 30e-6, 0e-6])),
        # Bead(np.array([0e-6, 40e-6, 0e-6])),
        # Bead(np.array([0, 51e-6, -45e-6])),
        Bead(np.array([0, 51e-6, 0e-6])),
        # Bead(np.array([0e-6, 60e-6, 0e-6])),
        # Bead(np.array([0e-6, 70e-6, 0e-6])),
        # Bead(np.array([0e-6, 80e-6, 0e-6])),
        # Bead(np.array([0e-6, 90e-6, 0e-6])),
        # Bead(np.array([0e-6, 99e-6, 0e-6])),
        # Bead(np.array([10e-6, 0, 0]))
    ]

    # nums = int(3e6)
    nums = np.array([1e4, 1e5, 1e6, 3e6, 6e6, 8e6, 1e7], dtype='int')
    # rays = np.array([])

    print(time.time() - start)
    for num in nums:
        start = time.time()
        rays = np.array([])
        for bead in beads:
            rays = np.append(rays, bead.generate_rays(fiber, num))

        if DRAW:
            bead.draw(fig1)

        final_positions = fiber.propagate(rays[0], fig1, draw=DRAW)
        for i in range(1, rays.size):
            final_positions = np.vstack((final_positions, fiber.propagate(rays[i], fig1, draw=DRAW)))
            if ((i - 1) * 100) % rays.size > (i * 100) % rays.size:
                print('progress: ', int((i / rays.size) / 0.01), '%')
        if DRAW:
            fig1.show()

        heatmap, xedges, yedges = np.histogram2d(final_positions[:, 0], final_positions[:, 1], bins=BIN_SIZE)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        # plt.imsave(getcwd() + '\\simulated_calibration\\' + 'img_y'+ str(bead.position[1]) + '_test_'+ str(nums) + '.tiff', heatmap.T)
        if DRAW:
            fig2.show()

        end = time.time()

        print("Nums:", num, "Simulation time: ", end - start)

