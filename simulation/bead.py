import math
import numpy as np
from simulation.ray import Ray


class Bead:

    def __init__(self, position=np.array([0, 0, 0]), radius=5e-6):
        self.position = position
        self.radius = radius

    def points_on_hemisphere(self, samples=10000):
        # TODO: standardize what samples refers to
        rnd = 1.
        offset = 2. / samples
        increment = math.pi * (3. - math.sqrt(5.))

        points = np.zeros((samples, 3))

        i = 0
        while i <= samples:
            # TODO: test if this is correct
            z = - (((i * offset) - self.radius) + (offset / 2))
            r = math.sqrt(1 - pow(z, 2))
            theta = ((i + rnd) % samples) * increment  # azimuthal (projection) angle
            psi = math.atan2(r, z)
            if psi >= np.pi() / 2:
                points = points[:i]
                break
            points[i] = np.array([self.position[0] + r * np.cos(theta),
                                  self.position[1] + r * np.sin(theta),
                                  self.position[2] + z])
        return points

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
        point_rays = np.array([Ray(self.position) for i in range(samples)])

        i = 0
        while i <= samples:
            z = - (((i * offset) - 1) + (offset / 2))
            r = math.sqrt(1 - pow(z, 2))
            theta = ((i + rnd) % samples) * increment  # azimuthal (projection) angle
            psi = math.atan2(r, z)
            if psi > psi_cutoff:  # zenith angle
                point_rays = point_rays[:i]
                break
            point_rays[i].psi = psi
            point_rays[i].theta = theta

            # TODO: move this condition inside refract
            if self.position[2] < 0:
                r = - point_rays[i].start[2] * np.tan(point_rays[i].psi)
                x = point_rays[i].start[0] + r * np.cos(point_rays[i].theta)
                y = point_rays[i].start[1] + r * np.sin(point_rays[i].theta)

                # assuming ellipse is centered at (0, 0)
                if (x ** 2 / a ** 2) + (y ** 2 / b ** 2) > 1:
                    i -= 1  # overwrite (essentially discard) this ray in next iteration
                    samples -= 1
            i += 1
        point_rays = point_rays[:i]  # discard all initialized rays after break index
        return point_rays

    # def generate_rays_on_hemisphere(self, fiber, samples):
    #     points = self.points_on_hemisphere(samples)
    #     bead_rays = np.array([])
    #     for point in points:
    #         point_rays = self.generate_rays(fiber, samples)
    #         bead_rays = np.hstack((rays, point_rays))
    #
    #     return bead_rays

    def draw(self, fig):
        pass
