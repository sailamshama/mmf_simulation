import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class Fiber:
    NA = 0.39  # Numerical Aperture
    core_index = 1.4630  # Refractive index 88nm RI.info
    cladding_index = 1.45
    surrounding_index = 1
    ellipse_a = 10e-6  # semi-major axis of fiber cross section
    ellipse_b = 100e-6  # semi-minor axis of mmf cross section
    length = 8000e-6  # length of fiber in microns. 8000 um = 8 mm):

    def draw(self, fig):
        # plot cylinder source: https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
        # spherical tutorial: https://stackoverflow.com/questions/36816537/spherical-coordinates-plot-in-matplotlib
        # cylindrical tutorial: https://matplotlib.org/3.1.0/gallery/mplot3d/voxels_torus.html

        theta = np.linspace(0, np.pi*2, 1000)# assuming 50 um max
        z = np.linspace(-50e-6, self.length, 1000)
        theta_grid, z_grid = np.meshgrid(theta, z)
        # wikipedia page on ellipse: https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center
        r_grid = self.ellipse_a * self.ellipse_b /(np.sqrt((self.ellipse_b * np.cos(theta_grid)) ** 2 + (self.ellipse_a * np.sin(theta_grid)) ** 2))
        x_grid = r_grid * np.cos(theta_grid)
        y_grid = r_grid * np.sin(theta_grid)

        ax = fig.gca(projection='3d')
        ax.plot_surface(x_grid, y_grid, z_grid, alpha = 0.1)

    def reflect(self, ray):
        intersection = np.array([0,0,0])
        # start point of reflected ray =  intersection point of previous point and wall of fiber
        # consider edge case top face of fiber

        return Ray([intersection], theta, psi)


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