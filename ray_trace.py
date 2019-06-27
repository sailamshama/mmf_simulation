import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as multip
from helpers import *


class mmf_fiber:
    NA = 0.39
    n = 1.4630  # 488nm RI.info
    r = 200 # 2 um
    a = 0.996 * r  # x axis radius
    b = 1 * r  # y axis radius
    length = 4000 #

def generate_rays(init_pos, fiber=mmf_fiber, mesh_density=50, num_rays=1000 ** 2):
    density_cap = 1000

    x0, y0, z0 = init_pos
    theta_max = np.arcsin(fiber.NA / fiber.n) #Snell's law
    r0 = -z0 * np.tan(theta_max)
    r = np.linspace(r0, 0, mesh_density, endpoint=False)
    r_space = r[0] - r[1]
    mesh = np.array([[0, 0, 0]])
    origin = init_pos[:2]

    for i in r:
        theta = np.linspace(0, 2 * np.pi, int(12 * mesh_density * i / r0), endpoint=False)
        t_space = theta[1] - theta[0]
        theta = theta + np.random.rand(len(theta)) * t_space - t_space / 2
        rmesh = i + np.random.rand(len(theta)) * r_space - r_space / 2
        thetamesh = theta
        xmesh = origin[0] + rmesh.flat * np.cos(thetamesh.flat)
        ymesh = origin[1] + rmesh.flat * np.sin(thetamesh.flat)
        zmesh = np.zeros(xmesh.shape)
        mesh = np.append(mesh, np.stack([xmesh, ymesh, zmesh], axis=1), axis=0)

    mesh = mesh[1:]
    in_fiber_mask = in_fiber(mesh[:, :2], mmf_fiber)
    n_iter1 = np.sum(in_fiber_mask)

    if n_iter1 > 0:
        refined_density = (float(num_rays) / n_iter1) ** 0.5 * mesh_density

        refined_density = min(refined_density, density_cap)

        r = np.linspace(r0, 0, refined_density, endpoint=False)
        r_space = r[0] - r[1]
        mesh = np.array([[0, 0, 0]])
        origin = init_pos[:2]

        for i in r:
            theta = np.linspace(0, 2 * np.pi, int(12 * refined_density * i / r0), endpoint=False)
            t_space = theta[1] - theta[0]
            theta = theta + np.random.rand(len(theta)) * t_space - t_space / 2
            rmesh = i + np.random.rand(len(theta)) * r_space - r_space / 2
            thetamesh = theta
            xmesh = origin[0] + rmesh.flat * np.cos(thetamesh.flat)
            ymesh = origin[1] + rmesh.flat * np.sin(thetamesh.flat)
            zmesh = np.zeros(xmesh.shape)
            mesh = np.append(mesh, np.stack([xmesh, ymesh, zmesh], axis=1), axis=0)
        mesh = mesh[1:]
        in_fiber_mask = in_fiber(mesh[:, :2], mmf_fiber)

    mesh = mesh[in_fiber_mask]
    vec = mesh - init_pos
    vec = norm_rays(vec)

    return mesh, vec


def generate_rays_mc(init_pos, fiber=mmf_fiber, num_rays=100 ** 2):
    num_ray_cap = 6e6

    x0, y0, z0 = init_pos
    theta_max = np.arcsin(fiber.NA / fiber.n)
    r0 = -z0 * np.tan(theta_max)
    mesh = np.array([[0, 0, 0]])
    origin = init_pos[:2]

    luckyx = np.random.rand(num_rays)
    luckyx = np.sqrt(luckyx)
    r = r0 * luckyx
    x = origin[0] + r * np.cos(theta)
    y = origin[1] + r * np.sin(theta)
    z = np.zeros(x.shape)
    mesh = np.stack([x, y, z], axis=1)

    in_fiber_mask = in_fiber(mesh[:, :2], mmf_fiber)
    n_iter1 = np.sum(in_fiber_mask.astype(int))
    mesh = mesh[in_fiber_mask]

    if n_iter1 > 0:
        n_iter2 = (num_rays - n_iter1) * float(num_rays) / n_iter1
        n_iter2 = int(n_iter2)

        n_iter2 = min(num_ray_cap, n_iter2)

        luckyx = np.random.rand(n_iter2)
        luckyx = np.sqrt(luckyx)
        r = r0 * luckyx
        theta = 2 * np.pi * np.random.rand(n_iter2)

        x = origin[0] + r * np.cos(theta)
        y = origin[1] + r * np.sin(theta)
        z = np.zeros(x.shape)
        mesh_iter2 = np.stack([x, y, z], axis=1)
        in_fiber_mask = in_fiber(mesh_iter2[:, :2], mmf_fiber)
        mesh_iter2 = mesh_iter2[in_fiber_mask]

    mesh = np.append(mesh, mesh_iter2, axis=0)

    vec = mesh - init_pos
    vec = norm_rays(vec)

    return mesh, vec


def propagate_ray(ray, ray_pos, final_pos, index_num, fiber=mmf_fiber):
    theta = np.arctan(np.sqrt(ray[0] ** 2 + ray[1] ** 2) / ray[2])
    z = 0
    tan_theta = np.tan(theta)
    pos = ray_pos
    xy_ray = ray[:2]
    while True:
        xy_ray = norm(xy_ray)

        xy_dist, pos_ref, xy_ref_ray = chord(pos, xy_ray)

        if xy_dist / tan_theta > (fiber.length - z):
            fp = pos + xy_ray * xy_dist * ((fiber.length - z) / (xy_dist / tan_theta))
            break
        else:
            z += xy_dist / tan_theta
            xy_ray = xy_ref_ray
            pos = pos_ref
    final_pos[index_num] = fp
    return


def propagate(rays, ray_pos, share_dict, index_num, fiber=mmf_fiber, trace=False):
    global num_processes
    theta = xyz_transform_theta(rays)
    final_pos = np.zeros(ray_pos.shape)
    if trace:
        plot_ellipse(fiber.a, fiber.b)
    for i, ray in enumerate(rays):
        z = 0
        tan_theta = np.tan(theta[i])
        pos = ray_pos[i]
        xy_ray = ray[:2]
        while True:
            xy_ray = norm(xy_ray)

            xy_dist, pos_ref, xy_ref_ray = chord(pos, xy_ray, mmf_fiber)

            if xy_dist / tan_theta > (fiber.length - z):
                share_dict[index_num[i]] = pos + xy_ray * xy_dist * ((fiber.length - z) / (xy_dist / tan_theta))
                if trace:
                    plt.plot([pos[0], final_pos[i, 0]], [pos[1], final_pos[i, 1]], 'b-')
                break
            else:
                z += xy_dist / tan_theta
                if trace:
                    plt.plot([pos[0], pos_ref[0]], [pos[1], pos_ref[1]], 'b-')
                xy_ray = xy_ref_ray
                pos = pos_ref
    return

def propagate_multithread(rays, ray_pos, fiber=mmf_fiber, trace=False):
    global num_processes
    theta = xyz_transform_theta(rays)
    final_pos = np.zeros(ray_pos.shape)

    rays_part = np.array_split(rays, num_processes)
    ray_pos_part = np.array_split(ray_pos, num_processes)
    index_part = np.array_split(range(len(rays)), num_processes)
    manager = multip.Manager()
    f_pos = manager.dict()
    jobs = []

    if __name__ == '__main__':
        for k in range(len(rays_part)):
            p = multip.Process(target=propagate, args=(rays_part[k], ray_pos_part[k], f_pos, index_part[k]))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()

    return np.array(f_pos.values())


if __name__ == '__main__':

    mmf_fiber = Fiber()
    num_processes = multip.cpu_count()

    xyz, rays = generate_rays([50e-6, 0e-6, -0.0001e-6], num_rays = 100)

    print('number of rays: ' + str(len(xyz)))

    f_pos = propagate_multithread(rays, xyz[:, :2])

    heatmap, xedges, yedges = np.histogram2d(f_pos[:, 0], f_pos[:, 1], bins=75)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
