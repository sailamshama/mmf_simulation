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
    length = 8000e-6  # length of fiber in microns. 8000 um = 8 mm):

    @staticmethod
    def draw():
        # draw cylinder
        # source: https://stackoverflow.com/questions/26989131/add-cylinder-to-plot

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x = np.linspace(-1, 1, 100)
        z = np.linspace(-2, 2, 100)
        xc, zc = np.meshgrid(x, z)
        yc = np.sqrt(1 - xc ** 2)
        r_stride = 20
        c_stride = 10
        ax.plot_surface(xc, yc, zc, alpha=0.2, rstride=r_stride, cstride=c_stride)
        ax.plot_surface(xc, -yc, zc, alpha=0.2, rstride=r_stride, cstride=c_stride)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()


class Ray:
    def __init__(self, start=np.array([0, 0, 0]), theta=0, psi=0):
        Ray.start = start
        Ray.theta = theta
        Ray.psi = psi

    def reflected(self, fiber):
        # start point of reflected ray =  intersection point of previous point and wall of fiber
        # consider edge case top face of fiber
        # return reflected Ray
        pass

    def draw(self, final_point):
        pass

