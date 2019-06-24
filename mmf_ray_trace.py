import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp


class Ellipse:

    def __init__(self, a, b):
        self.major = a
        self.minor = b
        self.focusL =
        self.focusR =

class Ray:

    def __init__(self, R = 0 , theta = 0, psi = 0):
        self.R = R
        self.theta = theta
        self.psi = psi


def save_image(histogram):
    pass

def plot_ellipse(Ellipse):

    pass

def generate_rays(init_positions, num_rays = 1000000):
    rays = np.ndarray( (num_rays,) , dtype = Ray )

    for i in xrange(num_rays):
        rays[i] = Ray()

    return rays


def propagate_rays():

    pass

def get_intersection():

    pass

def get_reflection():

    pass


def return_final_position():

    pass


if __name__ == "__main__":
    NA = input("Enter NA of multimode fiber: ") or 0.22
    MMF_LEN = input("Enter length of multimode fiber (in mm): ") or 8
    BEAD_RADIUS = input("Enter bead radius in mm: ") or 0.01
    NUM_BEADS = int(input("Enter number of beads: ") or 1)
    RAY_COUNT = int(input("Enter ray count: ") or 1000000)
    MAJOR = input("Enter elliptal mmf's core's semimajor axis (in mm): ") or 100
    MINOR = input("Enter elliptal mmf's core's semiminor axis (in mm): ") or 100
    SAVEDIR = input("Enter directory to save simulated PSFs: ") or './tmp/'
    CORE_REF_INDEX = input("Enter refractive index of mmf code: ") or 1.467287

    psiMAX = asin(NA)
    psiMAX = asin(1 / CORE_REF_INDEX * sin(psiMax))
    rMax = tan(psiMax) * MMF_LEN




