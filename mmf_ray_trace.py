import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp


def generate_rays():


def get_intersection():


if __name__ == "__main__":

    NA = input("Enter NA of multimode fiber: ") or 0.22
    MMF_LEN =  input("Enter length of multimode fiber (in mm): ") or 8
    BEAD_RADIUS = input("Enter bead radius in mm: ") or 0.01
    NUM_BEADS = int( input("Enter number of beads: ") or 1 )
    RAY_COUNT = int(input("Enter ray count: ") or 1000000 )
    MAJOR = input("Enter elliptal mmf's core's semimajor axis (in mm): " ) or 100
    MINOR = input("Enter elliptal mmf's core's semiminor axis (in mm): " ) or 100
    SAVEDIR = input("Enter directory to save simulated PSFs: ") or './tmp/'
    CORE_REF_INDEX = input("Enter refractive index of mmf code: ") or 1.467287f



