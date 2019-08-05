import math
import numpy as np
from simulation.ray import Ray


class Bead:

    def __init__(self, position=np.array([0, 0, 0]), radius=5e-6):
        self.position = position
        self.radius = radius

    # def points_on_hemisphere(center, samples=10000):
    #     rnd = 1.
    #     offset = 2. / samples
    #     increment = math.pi * (3. - math.sqrt(5.))
    #
    #     points = np.zeros((samples, 3))
    #
    #     i = 0
    #     while i <= samples:
    #         z = - (((i * offset) - 1) + (offset / 2))
    #         r = math.sqrt(1 - pow(z, 2))
    #         theta = ((i + rnd) % samples) * increment  # azimuthal (projection) angle
    #         psi = math.atan2(r, z)
    #         points[i] = np.array([r * np.cos(theta), r * np.sin(theta), z])
    #     return points

    # Use fibonacci sphere algorithm optimize uniform distribution of 'samples' number of points on spherical cap
    def generate_rays(self, fiber, samples=100000):
        a = fiber.ellipse_a
        b = fiber.ellipse_b

        # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html
        if self.position[2] < 0:
            psi_cutoff = np.pi / 2 - np.arcsin(fiber.cladding_index / fiber.core_index)  # pi/2 - theta_c
        else:
            psi_cutoff = np.arcsin(fiber.NA)
        # TODO: parallelize
        rnd = 1.
        offset = 2. / samples
        increment = math.pi * (3. - math.sqrt(5.))

        # TODO: optimize this
        rays = np.array([Ray(self.position) for i in range(samples)])

        i = 0
        while i <= samples:
            z = - (((i * offset) - 1) + (offset / 2))
            r = math.sqrt(1 - pow(z, 2))
            theta = ((i + rnd) % samples) * increment  # azimuthal (projection) angle
            psi = math.atan2(r, z)
            if psi > psi_cutoff:  # zenith angle
                rays = rays[:i]
                break
            rays[i].psi = psi
            rays[i].theta = theta

            # TODO: move this condition inside refract
            if self.position[2] < 0:
                r = - rays[i].start[2] * np.tan(rays[i].psi)
                x = rays[i].start[0] + r * np.cos(rays[i].theta)
                y = rays[i].start[1] + r * np.sin(rays[i].theta)

                # assuming ellipse is centered at (0, 0)
                if (x ** 2 / a ** 2) + (y ** 2 / b ** 2) > 1:
                    i -= 1  # overwrite (essentially discard) this ray in next iteration
                    samples -= 1
            i += 1
        rays = rays[:i]  # discard all initialized rays after break index
        return rays

    def draw(self, fig):
        pass
