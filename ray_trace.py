import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as multiP

pos = [0,0,0]

#principle of operation
#implementation
#application
#reliability

class mmf_fibre:
    NA = 0.39
    n = 1.4630 #488nm RI.info
    r = 100e-6
    a = 0.996*r #x axis radius
    b = 1*r #y axis radius
    length=8000e-6
    
def norm(vec):
    if np.sqrt(np.sum(vec**2))==0:
        return np.array([0,0])
    return vec/np.sqrt(np.sum(vec**2))

def partition(arr_like,size):
    assert type(size)==int
    temp_list = [[] for i in range(len(arr_like)/size+1)]
    for i,val in enumerate(arr_like):
        temp_list[i/size].append(val)
    return temp_list
    

def norm_rays(rays):
    return rays/np.sqrt(np.tile(np.sum(rays**2,axis=1),[1,1]).transpose())
    
def visualize_vec(vec):
    plt.plot([0,vec[0]],[0,vec[1]])
    plt.show()

def in_fibre(pos,fibre=mmf_fibre):
    a = fibre.a
    b = fibre.b
    r = np.sqrt(np.sum(pos**2,axis=1))
    # print r.shape
    theta = np.arctan(pos[:,0]/pos[:,1])
    #theta[0] = 0 #0,0 point
    r_ellip = a*b/np.sqrt(a**2*np.cos(theta)**2+b**2*np.sin(theta)**2)
    #print r_ellip
    #print pos
    #print r
    return r<r_ellip

def generate_rays(init_pos,fibre=mmf_fibre,mesh_density = 50, num_rays = 1000**2): 
    
    #num_rays ~ 6*mesh density**2
    #mesh density cap = 1000, ~6 million rays
    density_cap = 1000
    
    x0,y0,z0 = init_pos
    theta_max = np.arcsin(fibre.NA/fibre.n)
    r0 = -z0*np.tan(theta_max)
    r = np.linspace(r0,0,mesh_density,endpoint=False)
    r_space = r[0]-r[1]
    mesh = np.array([[0,0,0]])
    origin = init_pos[:2]
    for i in r:
        theta = np.linspace(0,2*np.pi,int(12*mesh_density*i/r0),endpoint=False)
        t_space = theta[1]-theta[0]
        theta = theta+np.random.rand(len(theta))*t_space-t_space/2
        rmesh = i+np.random.rand(len(theta))*r_space-r_space/2
        thetamesh = theta
        #rmesh,thetamesh = np.meshgrid(i,theta)
        xmesh = origin[0]+rmesh.flat*np.cos(thetamesh.flat)
        ymesh = origin[1]+rmesh.flat*np.sin(thetamesh.flat)
        zmesh = np.zeros(xmesh.shape)
        mesh = np.append(mesh,np.stack([xmesh,ymesh,zmesh],axis=1),axis=0)
    mesh = mesh[1:]
    in_fibre_mask = in_fibre(mesh[:,:2],mmf_fibre)
    n_iter1 = np.sum(in_fibre_mask)
    if n_iter1>0:
        refined_density = (float(num_rays)/n_iter1)**0.5*mesh_density
        
        refined_density = min(refined_density,density_cap)
        
        r = np.linspace(r0,0,refined_density,endpoint=False)
        r_space = r[0]-r[1]
        mesh = np.array([[0,0,0]])
        origin = init_pos[:2]
        for i in r:
            theta = np.linspace(0,2*np.pi,int(12*refined_density*i/r0),endpoint=False)
            t_space = theta[1]-theta[0]
            theta = theta+np.random.rand(len(theta))*t_space-t_space/2
            rmesh = i+np.random.rand(len(theta))*r_space-r_space/2
            thetamesh = theta
            #rmesh,thetamesh = np.meshgrid(i,theta)
            xmesh = origin[0]+rmesh.flat*np.cos(thetamesh.flat)
            ymesh = origin[1]+rmesh.flat*np.sin(thetamesh.flat)
            zmesh = np.zeros(xmesh.shape)
            mesh = np.append(mesh,np.stack([xmesh,ymesh,zmesh],axis=1),axis=0)
        mesh = mesh[1:]
        in_fibre_mask = in_fibre(mesh[:,:2],mmf_fibre)
    
    mesh = mesh[in_fibre_mask]
    vec = mesh-init_pos
    vec = norm_rays(vec)
    
    #refraction
    #n0 = 1.
    #vec[:,2] = vec[:,2]*fibre.n/n0
    
    vec = norm_rays(vec)

    return mesh,vec
    
def generate_rays_mc(init_pos,fibre=mmf_fibre,num_rays = 100**2):
    
    num_ray_cap = 6e6
    
    x0,y0,z0 = init_pos
    theta_max = np.arcsin(fibre.NA/fibre.n)
    r0 = -z0*np.tan(theta_max)
    mesh = np.array([[0,0,0]])
    origin = init_pos[:2]
    
    #lcuky rolls to generate random distribution with linear profile
    #x1 = r0*np.random.rand(num_rays)
    #x2 = r0*np.random.rand(num_rays)
    #x1mask = (x1>x2).astype(int)
    #x2mask = (x2>=x1).astype(int)
    #luckyx = x1mask*x1+x2mask*x2
    luckyx = np.random.rand(num_rays)
    luckyx = np.sqrt(luckyx)
    r = r0*luckyx
    #plt.hist(r,bins=100)
    theta = 2*np.pi*np.random.rand(num_rays)
    
    x = origin[0]+r*np.cos(theta)
    y = origin[1]+r*np.sin(theta)
    z = np.zeros(x.shape)
    mesh = np.stack([x,y,z],axis=1)

    in_fibre_mask = in_fibre(mesh[:,:2],mmf_fibre)
    n_iter1 = np.sum(in_fibre_mask.astype(int))
    mesh = mesh[in_fibre_mask]
    
    if n_iter1 > 0:
        n_iter2 = (num_rays-n_iter1)*float(num_rays)/n_iter1
        n_iter2 = int(n_iter2)
        
        n_iter2 = min(num_ray_cap,n_iter2)
        
        luckyx = np.random.rand(n_iter2)
        luckyx = np.sqrt(luckyx)
        r = r0*luckyx
        theta = 2*np.pi*np.random.rand(n_iter2)
        
        x = origin[0]+r*np.cos(theta)
        y = origin[1]+r*np.sin(theta)
        z = np.zeros(x.shape)
        mesh_iter2 = np.stack([x,y,z],axis=1)
        in_fibre_mask = in_fibre(mesh_iter2[:,:2],mmf_fibre)
        mesh_iter2 = mesh_iter2[in_fibre_mask] 

    mesh = np.append(mesh,mesh_iter2,axis=0)
    
    vec = mesh-init_pos
    vec = norm_rays(vec)

    #refraction
    #n0 = 1.
    #vec[:,2] = vec[:,2]*fibre.n/n0
    #vec = norm_rays(vec)
    
    return mesh,vec

def get_wall_vec(fibre,x,y):
    a = fibre.a
    b = fibre.b
    #if y==0:
     #   return [0,1]
    w = np.array([(y*a**2),(-x*b**2)])
    w = norm(w)
    return w

def reflect(w,d):
    d_prime = np.dot(w,d)*w*2-d
    d_prime = norm(d_prime)
    return d_prime

def xyz_transform_theta(rays):
    #returns angle w.r.t to fibre axis
    return np.arctan(np.sqrt(rays[:,0]**2+rays[:,1]**2)/rays[:,2])
    
def guided_rays(rays,fibre=mmf_fibre):
    theta = xyz_transform_theta(rays)
    filtered_rays = theta<np.arcsin(fibre.NA/fibre.n)
    return filtered_rays

def chord(pos,ray,fibre=mmf_fibre,trace=False):
    if np.isnan(ray).all():
        return 100.,pos,np.array([0,0])
    a = fibre.a
    b = fibre.b
    
    x0,y0 = pos
    #ray = reflect(get_wall_vec(fibre,x0,y0),ray[:2])
    
    xd,yd = ray[:2]
    #A*l**2+B*l+C = 0
    A = xd**2/a**2+yd**2/b**2
    B = 2*xd*x0/a**2+2*yd*y0/b**2
    C = x0**2/a**2+y0**2/b**2-1
    #select the positive solution
    lamb = (-B+np.sqrt(B**2-4*A*C))/(2*A)
    
    # if trace:
    #     plt.plot([x0,x0+lamb*xd],[y0,y0+lamb*yd],'b*-')
    ref_pos = [x0+lamb*xd,y0+lamb*yd]
    ray = reflect(get_wall_vec(fibre,x0+lamb*xd,y0+lamb*yd),ray[:2])
    #print ray
    #print get_wall_vec(fibre,x0+lamb*xd,y0+lamb*yd)
    #print x0+lamb*xd,y0+lamb*yd
    # if trace:
    #     plt.plot([x0+lamb*xd,x0+lamb*xd+lamb*ray[0]],[y0+lamb*yd,y0+lamb*yd+lamb*ray[1]],'r*-')
    # 
    #print lamb
    #return length of chord, location of reflection and new direction
    return lamb, ref_pos, ray

def propagate_ray(ray,ray_pos,final_pos,index_num,fibre=mmf_fibre):
    theta = np.arctan(np.sqrt(ray[0]**2+ray[1]**2)/ray[2])
    z = 0
    tan_theta = np.tan(theta)
    pos = ray_pos
    xy_ray = ray[:2]    
    while True:
            xy_ray = norm(xy_ray)
            
            xy_dist,pos_ref,xy_ref_ray = chord(pos,xy_ray)
            
            if xy_dist/tan_theta>(fibre.length-z):
                fp = pos + xy_ray*xy_dist*((fibre.length-z)/(xy_dist/tan_theta))
                break
            else:
                z += xy_dist/tan_theta
                xy_ray = xy_ref_ray
                pos = pos_ref
    #print index_num
    final_pos[index_num] = fp 
    return

def propagate(rays,ray_pos,share_dict,index_num,fibre=mmf_fibre,trace=False):
    global num_threads
    theta = xyz_transform_theta(rays)
    final_pos = np.zeros(ray_pos.shape)
    if trace:
        plot_ellipse(fibre.a,fibre.b)
    for i,ray in enumerate(rays):
        z = 0
        tan_theta = np.tan(theta[i])
        pos = ray_pos[i]
        xy_ray = ray[:2]
        while True:
            xy_ray = norm(xy_ray)
            
            xy_dist,pos_ref,xy_ref_ray = chord(pos,xy_ray,trace=trace)
            
            
            if xy_dist/tan_theta>(fibre.length-z):
                share_dict[index_num[i]] = pos + xy_ray*xy_dist*((fibre.length-z)/(xy_dist/tan_theta))
                if trace:
                    plt.plot([pos[0],final_pos[i,0]],[pos[1],final_pos[i,1]],'b-')
                break
            else:
                z += xy_dist/tan_theta
                if trace:
                    plt.plot([pos[0],pos_ref[0]],[pos[1],pos_ref[1]],'b-')
                xy_ray = xy_ref_ray
                pos = pos_ref
    return
        #print 'ray completed!'
        #while pos[2]<fibre.length:

def propagate_multithread(rays,ray_pos,fibre=mmf_fibre,trace=False):
    global num_threads
    theta = xyz_transform_theta(rays)
    final_pos = np.zeros(ray_pos.shape)
    
    rays_part = np.array_split(rays,num_threads)
    ray_pos_part = np.array_split(ray_pos,num_threads)
    index_part = np.array_split(range(len(rays)),num_threads)
    manager = multiP.Manager()
    f_pos = manager.dict()
    
    jobs = []
    if __name__ == '__main__':
        for k in range(len(rays_part)):
            p = multiP.Process(target = propagate,args=(rays_part[k],ray_pos_part[k],f_pos,index_part[k]))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
    return np.array(f_pos.values())
    
def plot_ellipse(a,b):
    theta = np.linspace(0,2*np.pi,1001)
    r = a*b/(np.sqrt((b*np.cos(theta))**2+(a*np.sin(theta))**2))
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    plt.plot(x,y)


if __name__=='__main__':
    num_threads=6
    
    #MC - monte carlo generation (fully random)
    #    - has adaptive sizing for approximately same number of rays despite incomplete overlap 
    
    #xyz,rays = generate_rays_mc([50e-6,50e-6,-0.0001e-6],num_rays=962301)
    
    #normal - psudorandom generation 
    #    - has adaptive sizing for approximately same number of rays despite incomplete overlap 
    
    xyz,rays = generate_rays([50e-6,0e-6,-0.0001e-6],num_rays=100)
    
    # print 'number of rays: '+str(len(xyz))
    f_pos=propagate_multithread(rays,xyz[:,:2])
    heatmap, xedges, yedges = np.histogram2d(f_pos[:,0], f_pos[:,1], bins=75)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    
    
    
