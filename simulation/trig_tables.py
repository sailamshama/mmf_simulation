import numpy as np


# TODO: use appropriate structure
class TrigTable:
    # define precision

    angles = np.linspace(0, np.pi * 2, 100000)
    sines = np.sin(angles)
    cosines = np.cos(angles)
    tans = np.tan(angles)

    sin_dict = dict(zip(angles, sines))
    cos_dict = dict(zip(angles, cosines))
    tan_dict = dict(zip(angles, tans))

    # TODO implement efficient algorithm
    def sin(self, angle):
        nearest_angle_index = (np.abs(self.angles - np.repeat(angle, len(self.angles)))).argmin()
        return self.sines[nearest_angle_index]

    def cos(self, angle):
        nearest_angle_index = (np.abs(self.angles - np.repeat(angle, len(self.angles)))).argmin()
        return self.cosines[nearest_angle_index]

    def tan(self, angle):
        nearest_angle_index = (np.abs(self.angles - np.repeat(angle, len(self.angles)))).argmin()
        return self.tans[nearest_angle_index]