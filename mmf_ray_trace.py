import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

# NA = PApplet.parseFloat(readInputLine());
# len = PApplet.parseFloat(readInputLine());
# beadX = PApplet.parseFloat(readInputLine());
# beadY = PApplet.parseFloat(readInputLine());
# beadRad = PApplet.parseFloat(readInputLine());
# rayMax = PApplet.parseInt(readInputLine());
# pixType = PApplet.parseInt(readInputLine());
# major = PApplet.parseFloat(readInputLine());
# minor = PApplet.parseFloat(readInputLine());
# save director






if __name__ == "__main__":

    NA = input("Enter NA of multimode fiber: ") or 0.22
    MMF_LEN =  input("Enter length of multimode fiber (in mm): ") or 8
    BEAD_RADIUS = input("Enter bead radius in mm: ") or 0.01
    NUM_BEADS = int( input("Enter number of beads: ") or 1 )
    RAY_COUNT = int(input("Enter ray count: ") or 1000000 )
    MAJOR = input("Enter elliptal mmf's core's semimajor axis (in mm): " ) or 100
    MINOR = input("Enter elliptal mmf's core's semiminor axis (in mm): " ) or 100
    SAVEDIR = input("Enter directory to save simulated PSFs: ") or './tmp/'


