import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


class Fiber:
    NA = 0.39  # Numerical Aperture
    core_index = 1.4630  # Refractive index 88nm RI.info
    cladding_index = 1.45
    surrounding_index = 1
    ellipse_a = 100e-6  # semi-major axis of fiber cross section
    ellipse_b = 100e-6  # semi-minor axis of mmf cross section
    length = 12000e-6   # length of fiber in microns. 8000 um = 8 mm):

    # plot cylinder source: https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
    # spherical tutorial: https://stackoverflow.com/questions/36816537/spherical-coordinates-plot-in-matplotlib
    # cylindrical tutorial: https://matplotlib.org/3.1.0/gallery/mplot3d/voxels_torus.html

    theta = np.linspace(0, np.pi * 2, 1000)  # assuming 50 um max
    z = np.linspace(0, length, 1000)
    theta_grid, z_grid = np.meshgrid(theta, z)
    # wikipedia page on ellipse: https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center
    r_grid = ellipse_a * ellipse_b / (np.sqrt((ellipse_b * np.cos(theta_grid)) ** 2 + (ellipse_a * np.sin(theta_grid)) ** 2))
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    def draw(self, fig):

        ax = fig.gca(projection='3d')
        ax.plot_surface(self.x_grid, self.y_grid, self.z_grid, alpha=0.1)

        ax.set(xlim=(-1.5 * self.ellipse_a, 1.5 * self.ellipse_a), ylim=(-1.5 * self.ellipse_b, 1.5 * self.ellipse_b), zlim=(-100e-6, 1.2 * self.length))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def get_intersection(self, ray):
        # TODO: debug this
        # start point of reflected ray =  intersection point of previous point and wall of fiber
        # consider edge case top face of fiber and bottom face of fiber
        # refer to mathematica notebook intersection.nb
        # x, y = position of ray

        a = self.ellipse_a
        b = self.ellipse_b
        t = ray.theta
        xi = ray.start[0]
        yi = ray.start[1]

        intersection = np.array([0,0,0])
        intersection[0] = (b * np.cos(t)) **2 + (a * np.sin(t)) ** 2
        intersection[0] += (np.sin(t) * xi - np.cos(t) * yi) ** 2
        intersection[0] *= (a*b)**2
        intersection[0] = np.sqrt(intersection[0])
        intersection[0] = - (b**2) * np.cos(t) * xi
        intersection[0] += - (a**2) * np.sin(t) * yi
        intersection[0] /= ((b * np.cos(t)) ** 2 + (a * np.sin(t)) ** 2)
        intersection[0] *= np.cos(t)
        intersection[0] += xi

        intersection[1] = (b * np.cos(t)) **2 + (a * np.sin(t)) ** 2
        intersection[1] += (np.sin(t) * xi - np.cos(t) * yi) ** 2
        intersection[1] *= (a*b)**2
        intersection[1] = np.sqrt(intersection[0])
        intersection[1] = - (b**2) * np.cos(t) * xi
        intersection[1] += - (a**2) * np.sin(t) * yi
        intersection[1] /= ((b * np.cos(t)) ** 2 + (a * np.sin(t)) ** 2)
        intersection[1] *= np.sin(t)
        intersection[1] += yi

        intersection[2] = np.sqrt((intersection[0] - xi) ** 2 + (intersection[1] - yi) ** 2) * np.tan(ray.psi)

        return intersection

    def reflect(self, ray):
        # TODO: test this
        intersection = self.get_intersection(ray)
        ray_length = np.sqrt(np.sum(ray.start ** 2))
        reflection_angle = np.arcsin((intersection[2] - ray.start[2]) / ray_length)
        psi_reflected_ray = np.pi / 2 - reflection_angle
        reflected_ray = Ray(start=intersection, theta=ray.theta, psi=psi_reflected_ray)
        return reflected_ray

class Ray:
    def __init__(self, start=np.array([0, 0, 0]), theta=0, psi=0):
        Ray.start = start
        Ray.theta = theta
        Ray.psi = psi

    def draw(self, fig, final_point):
        xs = np.array([self.start[0], final_point[0]])
        ys = np.array([self.start[1], final_point[1]])
        zs = np.array([self.start[2], final_point[2]])

        ax = fig.gca(projection = '3d')
        ax.plot(xs,ys,zs)
