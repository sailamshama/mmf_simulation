import numpy as np


def get_wall_vec(fiber, x, y):
    a = fiber.a
    b = fiber.b
    w = np.array([(y * a ** 2), (-x * b ** 2)])
    w = norm(w)

    return w

def reflect(w, d):
    d_prime = np.dot(w, d) * w * 2 - d
    d_prime = norm(d_prime)

    return d_prime

def xyz_transform_theta(rays):
    # returns angle w.r.t to fiber axis

    return np.arctan(np.sqrt(rays[:, 0] ** 2 + rays[:, 1] ** 2) / rays[:, 2])

def guided_rays(rays, fiber):
    theta = xyz_transform_theta(rays)
    filtered_rays = theta < np.arcsin(fiber.NA / fiber.n)

    return filtered_rays

def norm(vec):
    if np.sqrt(np.sum(vec ** 2)) == 0:
        return np.array([0, 0])

    return vec / np.sqrt(np.sum(vec ** 2))

def partition(arr_like, size):
    assert type(size) == int
    temp_list = [[] in range(len(arr_like) / size + 1)]
    for i, val in enumerate(arr_like):
        temp_list[i / size].append(val)
    return temp_list

def norm_rays(rays):
    return rays / np.sqrt(np.tile(np.sum(rays ** 2, axis=1), [1, 1]).transpose())

def visualize_vec(vec):
    plt.draw([0, vec[0]], [0, vec[1]])
    plt.show()

def in_fiber(pos, fiber):
    a = fiber.a
    b = fiber.b
    r = np.sqrt(np.sum(pos ** 2, axis=1))
    theta = np.arctan(pos[:, 0] / pos[:, 1])
    r_ellip = a * b / np.sqrt(a ** 2 * np.cos(theta) ** 2 + b ** 2 * np.sin(theta) ** 2)

    return r < r_ellip

def chord(pos, ray, fiber):
    if np.isnan(ray).all():
        return 100., pos, np.array([0, 0])
    a = fiber.a
    b = fiber.b

    x0, y0 = pos
    xd, yd = ray[:2]

    A = xd ** 2 / a ** 2 + yd ** 2 / b ** 2
    B = 2 * xd * x0 / a ** 2 + 2 * yd * y0 / b ** 2
    C = x0 ** 2 / a ** 2 + y0 ** 2 / b ** 2 - 1

    lamb = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    ref_pos = [x0 + lamb * xd, y0 + lamb * yd]
    ray = reflect(get_wall_vec(fiber, x0 + lamb * xd, y0 + lamb * yd), ray[:2])

    return lamb, ref_pos, ray

def plot_ellipse(a, b):

    theta = np.linspace(0, 2 * np.pi, 1001)
    r = a * b / (np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.draw(x, y)

