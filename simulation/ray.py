import vectormath as vmath
import numpy as np

class Fiber:
    NA = 0.39  # Numerical Aperature
    core_index = 1.4630  # Refractive index 88nm RI.info
    cladding_index = 1.45
    surrounding_index = 1
    ellipse_a = 100e-6  # semi-major axis of fiber cross section
    ellipse_b = 100e-6  # semi-minor axis of mmf cross section
    length = 8000e-6  # length of fiber in microns. 8000 um = 8 mm


class Ray():
    def __init__(self, start = np.array([0,0]), theta = 0, psi = 0):
        Ray.start = start
        Ray.theta = theta
        Ray.psi = psi
