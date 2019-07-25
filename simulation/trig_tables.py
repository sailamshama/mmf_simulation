import numpy as np

ANGLE_PRECISION = 0.001


class TrigTable:
    # define precision

    angles = np.arange(0, np.pi * 2, ANGLE_PRECISION)
    count = np.shape(angles)[0]

    sines = np.sin(angles)
    cosines = np.cos(angles)
    tans = np.tan(angles)

    @staticmethod
    def sin(angle):
        return TrigTable.sines[int(angle / ANGLE_PRECISION) % TrigTable.count]

    @staticmethod
    def cos(angle):
        return TrigTable.cosines[int(angle / ANGLE_PRECISION) % TrigTable.count]

    @staticmethod
    def tan(angle):
        return TrigTable.tans[int(angle / ANGLE_PRECISION) % TrigTable.count]


if __name__ == '__main__':
    import time

    start = time.time()
    for i in range(100):
        np.sin(i)

    end = time.time()
    print(end-start)
    start = time.time()
    for i in range(100):
        TrigTable.sin(i)
    end = time.time()
    print(end-start)

