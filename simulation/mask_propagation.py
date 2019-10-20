import numpy as np
import math
import matplotlib.pyplot as plt
from simulation.fiber import Fiber
from simulation.bead import Bead
import time
from matplotlib.patches import Ellipse

start = time.time()

SAMPLES = int(1e7)
BIN_SIZE = 150

FIBER_DISTANCE = 50e-6
OBJECT_DISTANCE = 50e-6
HOLE_SIZE = 0.5e-6
p = 101

np.random.seed(10)
N = np.random.random((p, p))
M = N > 0.7

M = np.ones((p, p))
for i in np.linspace(0, 90, 10):
    j = int(i)
    M[j:j+5, :] = 0

fiber = Fiber()
bead = Bead(np.array([-20e-6/np.sqrt(2), 20e-6/np.sqrt(2), -FIBER_DISTANCE - OBJECT_DISTANCE]))
rays = bead.generate_rays(fiber, samples=SAMPLES)
propagated_rays = []

i = 0
for ray in rays:
    mura_pos = np.array([0, 0, -FIBER_DISTANCE])
    mura_pos[2] = -FIBER_DISTANCE
    mura_pos[1] = ray.start[1] + OBJECT_DISTANCE * np.tan(ray.psi) * np.sin(ray.theta)
    mura_pos[0] = ray.start[0] + OBJECT_DISTANCE * np.tan(ray.psi) * np.cos(ray.theta)

    # TODO: check this
    Mi = math.floor((mura_pos[0] + math.floor(p/2) * HOLE_SIZE) / HOLE_SIZE)
    Mj = math.floor((mura_pos[1] + math.floor(p/2) * HOLE_SIZE) / HOLE_SIZE)

    if Mi > 100 or Mj > 100:
        continue

    if not M[Mi, Mj]:
        propagated_rays.append(ray)

prays = np.array([propagated_rays[i] for i in range(len(propagated_rays))])

fig1 = plt.figure(figsize=(5,10))

final_positions = fiber.propagate(prays[0], fig1, draw=False)
for i in range(1, prays.size):
    final_positions = np.vstack((final_positions, fiber.propagate(prays[i], fig1, draw=False)))
    if ((i - 1) * 100) % prays.size > (i * 100) % prays.size:
        print('progress: ', int((i / prays.size) / 0.01), '%')

simulation_time = time.time() - start

plt.subplot(3, 1, 1, aspect='equal')
ax = plt.gca()
ellipse = Ellipse(xy = (0, 0),
        width = 200e-6,
        height=200e-6,
        angle=0,
        edgecolor='black',
        facecolor='white')
ax.add_patch(ellipse)
plt.plot(50e-6, 50e-6, 'o', color='green')
plt.xlim(-110e-6, 110e-6)
plt.ylim(-110e-6, 110e-6)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.title("Point source location: ( %f , %f)" % (bead.position[0], bead.position[1]))

plt.subplot(3, 1, 2)
plt.imshow(M)
plt.title("Mask")


plt.subplot(3, 1, 3)
heatmap, xedges, yedges = np.histogram2d(final_positions[:, 0], final_positions[:, 1], bins=BIN_SIZE)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.title("Simulation time: %2.2f min" % (simulation_time/60))

fig1.show()
