import numpy as np
from simulation.ray import Ray
import mpl_toolkits.mplot3d

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
        l = self.length
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.x_grid, self.y_grid, self.z_grid, alpha=0.1)
        ax.set(xlim=(-1.2 * a, 1.2 * a), ylim=(-1.5 * b, 1.5 * b), zlim=(-100e-6, 1.2 * l))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=0, azim=90)

    def get_intersection(self, ray, fig, draw):

        if ray.psi == 0:
            return np.array([ray.start[0], ray.start[1], self.length])

        if ray.start[2] < 0:
            r = - ray.start[2] * np.tan(ray.psi)
            x = ray.start[0] + r * np.cos(ray.theta)
            y = ray.start[1] + r * np.sin(ray.theta)
            intersection = np.array([x, y, 0])
            if draw:
                ray.draw(fig, intersection)
            return intersection

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

        if draw:
            if intersection[2] <= self.length:
                ray.draw(fig, intersection)

        return intersection

    def reflect(self, ray, fig, draw=False):

        if ray.start[2] == self.length:
            return ray
        intersection = self.get_intersection(ray, fig, draw)

        ray_length = np.sqrt(np.sum((intersection - ray.start) ** 2))
        vertical_reflection_angle = np.arcsin((intersection[2] - ray.start[2]) / ray_length)
        psi_reflected_ray = np.pi / 2 - vertical_reflection_angle

        theta_x = np.arctan2(-self.ellipse_b**2 * intersection[0], self.ellipse_a**2 * intersection[1])
        theta_reflected_ray = 2*theta_x - ray.theta

        reflected_ray = Ray(start=intersection, theta=theta_reflected_ray, psi=psi_reflected_ray)
        return reflected_ray

    def refract(self, ray, fig, draw=False):

        # TODO: case when rays are outside fiber
        # if (abs(ray.start[0]) > self.ellipse_a) or (abs(ray.start[1]) > self.ellipse_b):
        #     pass

        intersection = self.get_intersection(ray, fig, draw)
        refracted_ray = Ray(np.array([intersection[0], intersection[1], intersection[2]]))

        refracted_ray.psi = np.arcsin(self.surrounding_index / self.core_index * np.sin(ray.psi))
        refracted_ray.theta = ray.theta

        return refracted_ray

    def propagate(self, ray, fig, draw=False, log_lengths=False):
        # TODO: debug log lengths
        if log_lengths:
            lengths = np.array([])

        if ray.start[2] < 0:
            reflected_ray = self.refract(ray, fig, draw)
            if log_lengths:
                lengths = np.append(lengths, np.sqrt((reflected_ray.start - ray.start)**2))
        else:
            reflected_ray = ray
        i = 0

        while reflected_ray.start[2] < self.length:  # and (i < 20):
            ray = reflected_ray
            reflected_ray = self.reflect(ray, fig, draw)
            if log_lengths:
                lengths = np.append(lengths, np.sqrt((reflected_ray.start - ray.start)**2))
            i += 1

        if log_lengths:
            return (reflected_ray.start, lengths)

        else:
            return reflected_ray.start

    # def probe_lengths():
    #     # plot histogram of lengths of rays reflected within the fiber to see if there's a pattern
    #
    #     fiber = Fiber()
    #     fig = plt.figure()
    #     fiber.draw(fig)
    #
    #     psi_max = np.pi / 2 - np.arcsin(fiber.cladding_index / fiber.core_index)
    #     sample_ray = Ray(np.array([0, 0, -10e-6]), 0, psi_max)
    #     final_pos, lengths = fiber.propagate(sample_ray, fig, draw=True, log_lengths=True)
    #     plt.show()
