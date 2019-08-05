import numpy as np


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
