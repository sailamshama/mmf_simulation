import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math
import random


class Fiber:
    NA = 0.39             # Numerical Aperture
    core_index = 1.4589
    cladding_index = 1.398200
    surrounding_index = 1
    # TODO: adjust this to core radius
    ellipse_a = 100e-6    # semi-major axis of fiber cross section
    ellipse_b = 100e-6    # semi-minor axis of mmf cross section
    length = 12500e-6     # length of fiber in microns. 8000 um = 8 mm):

    # plot cylinder source: https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
    # spherical tutorial: https://stackoverflow.com/questions/36816537/spherical-coordinates-plot-in-matplotlib
    # cylindrical tutorial: https://matplotlib.org/3.1.0/gallery/mplot3d/voxels_torus.html

    theta = np.linspace(0, np.pi * 2, 1000)  # assuming 50 um max
    z = np.linspace(0, length, 1000)
    theta_grid, z_grid = np.meshgrid(theta, z)
    # wikipedia page on ellipse: https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center
    r_grid = ellipse_a*ellipse_b / (np.sqrt((ellipse_b * np.cos(theta_grid)) ** 2 + (ellipse_a*np.sin(theta_grid))**2))
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    def draw(self, fig):
        a = self.ellipse_a
        b = self.ellipse_b
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.x_grid, self.y_grid, self.z_grid, alpha=0.1)
        ax.set(xlim=(-1.5 * a, 1.5 * a), ylim=(-1.5 * b, 1.5 * b), zlim=(-500e-6, 0.1 * self.length))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=0, azim=90)

    def get_intersection(self, ray):

        if ray.psi == 0:
            return np.array([ray.start[0], ray.start[1], self.length])

        if ray.start[2] < 0:
            r = - ray.start[2] * np.tan(ray.psi)
            x = ray.start[0] + r * np.cos(ray.theta)
            y = ray.start[1] + r * np.sin(ray.theta)

            return np.array([x, y, 0])

        a = self.ellipse_a
        b = self.ellipse_b
        t = ray.theta
        xi = ray.start[0]
        yi = ray.start[1]
        zi = ray.start[2]

        # refer to mathematica notebook intersection.nb
        intersection = np.zeros([3])
        temp1 = ((b * np.cos(t)) ** 2 + (a * np.sin(t)) ** 2)
        temp2 = (a**2 * b**2) * (temp1 - (np.sin(t) * xi - np.cos(t) * yi) ** 2)
        if temp2 < 0:
            temp3 = 0
        else:
            temp3 = np.sqrt(temp2)
        temp4 = (-b**2 * np.cos(t) * xi - a**2 * np.sin(t) * yi + temp3) / temp1

        intersection[0] = xi + temp4 * np.cos(t)
        intersection[1] = yi + temp4 * np.sin(t)
        intersection[2] = zi + np.sqrt((intersection[0] - xi) ** 2 + (intersection[1] - yi) ** 2) / np.tan(ray.psi)

        # last ray case: similiar triangles argument
        if intersection[2] > self.length:
            r = (self.length - zi) / (intersection[2] - zi)
            intersection[0] = (intersection[0] - xi) * r + xi
            intersection[1] = (intersection[1] - yi) * r + yi
            intersection[2] = self.length

        return intersection

    def reflect(self, ray):

        if ray.start[2] == self.length:
            return ray
        intersection = self.get_intersection(ray)
        ray_length = np.sqrt(np.sum((intersection - ray.start) ** 2))
        vertical_reflection_angle = np.arcsin((intersection[2] - ray.start[2]) / ray_length)
        psi_reflected_ray = np.pi / 2 - vertical_reflection_angle

        theta_x = np.arctan2(-self.ellipse_b**2 * intersection[0], self.ellipse_a**2 * intersection[1])
        theta_reflected_ray = 2*theta_x - ray.theta

        reflected_ray = Ray(start=intersection, theta=theta_reflected_ray, psi=psi_reflected_ray)
        return reflected_ray

    def refract(self, ray):
        # check that ray.start[2] == 0
        intersection = self.get_intersection(ray)
        refracted_ray = Ray(np.array([intersection[0], intersection[1], intersection[2]]))

        refracted_ray.psi = np.arcsin(self.surrounding_index / self.core_index * np.sin(ray.psi))
        refracted_ray.theta = ray.theta
        return refracted_ray

    def propagate(self, ray, fig, draw=False):

        if ray.start[2] < 0:
            reflected_ray = self.refract(ray)
        else:
            reflected_ray = ray
        i = 0
        while reflected_ray.start[2] < self.length:  # and (i < 20):
            ray = reflected_ray
            if draw:
                ray.draw(fig, self.get_intersection(reflected_ray))
            reflected_ray = self.reflect(ray)
            i += 1

        return reflected_ray.start


class Ray:
    # TODO: throw errors for invalid values
    def __init__(self, start=np.array([0, 0, 0]), theta=0, psi=0):
        self.start = start
        self.theta = theta
        self.psi = psi

    def draw(self, fig, final_point):
        xs = np.array([self.start[0], final_point[0]])
        ys = np.array([self.start[1], final_point[1]])
        zs = np.array([self.start[2], final_point[2]])

        ax = fig.gca(projection='3d')
        ax.plot(xs, ys, zs)
        # TODO: live plot rays
        # plt.pause(0.05)
        # plt.close()
        # plt.show()

    def length(self, final_point):
        return np.sqrt(sum((final_point - self.start) ** 2))


class Bead:

    def __init__(self, radius=5e-6, position=np.array([0, 0, 0])):
        self.position = position
        self.radius = radius

    def generate_rays(self, psi_cutoff=math.pi, samples=100000):
        # TODO: parallelize
        r = self.radius

        rnd = 1.
        offset = 2. / samples
        increment = math.pi * (3. - math.sqrt(5.))

        # TODO: optimize this
        rays = np.array([Ray(self.position) for i in range(samples)])

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

            while np.sqrt(sum(rays[:2]**2)) <= r:
                rays[i].start[0] = self.position[0] + random.uniform(-r, r)
                rays[i].start[1] = self.position[0] + random.uniform(-r, r)

        return rays

    def draw(self, fig):
        # TODO: draw rays out of bead and hitting bottom of fiber (assume ideal point size)
        # TODO: draw rays post-refraction
        # TODO: account for bead size (radius)
        # TODO: implement finite bead size
        # TODO: implement period rays acceleration
        # TODO: accelerate on GPU
        ax = fig.gca(projection='3d')


###### TESTS ######


def test_refract():
    fiber = Fiber()
    fig = plt.figure()
    fiber.draw(fig)

    init_points = np.array([
        # [0e-6, 0e-6, 0e-6],
        [0, 50e-6, -50e-6],
        # [75e-6, 0, 0],
        # [10e-6, 0, 0]
    ])

    # TODO: find a better way to store psi_max
    # https://circuitglobe.com/numerical-aperture-of-optical-fiber.html
    # psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index)
    psi_max = np.arcsin(fiber.NA)
    psi_max = np.repeat(psi_max, init_points.shape[0])
    for i, init_point in enumerate(init_points):
        if init_point[2] == 0:
            psi_max[i] = np.pi / 2 - np.arcsin(fiber.cladding_index / fiber.core_index)  # pi/2 - theta_c

    nums = int(3e3)
    generated_rays = generate_rays_multiple_sources(init_points, bead_sizes, psi_max, nums)


    sample_ray = Ray(np.array([0, 0, -300e-6]), 0, np.pi/20)
    refracted_ray = fiber.refract(sample_ray)
    sample_ray.draw(fig, refracted_ray.start)
    reflected_ray = fiber.reflect(refracted_ray)
    refracted_ray.draw(fig, reflected_ray.start)

    fiber.propagate(refracted_ray, fig, draw=True)

    plt.show()



def test_bead():
    fiber = Fiber()
    fig = plt.figure()
    fiber.draw(fig)

    init_points = np.array([
        # [0e-6, 0e-6, 0e-6],
        [0, 50e-6, -50e-6],
        # [75e-6, 0, 0],
        # [10e-6, 0, 0]
    ])

    BEAD_SIZE = 2e-3
    bead_sizes = np.repeat(BEAD_SIZE, init_points.shape[0])
    psi_max = np.arcsin(fiber.surrounding_index * fiber.NA / fiber.core_index)


    pass

def test_psi():
    fiber = Fiber()
    psi_out = np.arcsin(fiber.NA)
    psi_in = np.pi/2 - np.arcsin(fiber.cladding_index / fiber.core_index)

    return psi_in < psi_out

if __name__ == "__main__":
    BEAD_SIZE = 2e-3


    # test_refract()
