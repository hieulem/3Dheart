import torch
from data.data import normalize_vertices, sample_outer_surface_in_voxel, voxel2mesh, clean_border_pixels, normalize_vertices2
import numpy as np
import torch.nn.functional as F
from torchmcubes import marching_cubes # https://github.com/tatsy/torchmcubes
from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
from utils.line_intersect import doIntersect
from IPython import embed

def voxel_surface_sampler(y, i, point_count=3000):

    shape = torch.tensor(y.shape)[None].float()
    if i == 1: 
        y_outer = sample_outer_surface_in_voxel((y==i).long(), inner=False)  
    else:
        y_outer = sample_outer_surface_in_voxel((y>0).long(), inner=False) 

    surface_points2 = torch.nonzero(y_outer)
    surface_points2 = normalize_vertices2(surface_points2, shape.cuda()) 
    surface_points = torch.flip(surface_points2, dims=[1]).float()  # convert z,y,x -> x, y, z 

    perm = torch.randperm(len(surface_points))  
    surface_points = surface_points[perm[:np.min([len(perm), point_count])]]

    return surface_points

def voxel_marching_cube(y, i):
    # no sampler for this (we don't use it)

    shape = torch.tensor(y.shape)[None].float() 
    gap = 1
    y_ = clean_border_pixels((y==i).long(), gap=gap)
    vertices_mc, faces_mc = voxel2mesh(y_, gap, shape)  

    return vertices_mc, faces_mc


def contour_sample(x, y, contours, i, resolution, point_count=3000): 
    contour = contours[i]
    surface_points = {} 

    for slice_id, contour_2d in contour.items():    
        surface_points[slice_id] = contour_2d

    return surface_points

def contour_sample2(x, y, contours, i, point_count=3000):
    shape = torch.tensor(y.shape).flip([0])[None].float()
    contour = contours[i]
    surface_points = []

    # xs_mesh = []
    # _, _, H, W = x.shape
    # x_ = torch.linspace(0, W-1, steps=W)
    # y_ = torch.linspace(0, H-1, steps=H)
    # grid_x, grid_y = torch.meshgrid(x_, y_, indexing='ij')
    # grid_x = grid_x.reshape(-1)[:, None]
    # grid_y = grid_y.reshape(-1)[:, None]
    # grid = torch.cat([grid_x, grid_y], dim=1)

    for slice_id, contour_2d in contour.items():    
        if len(contour_2d.shape) == 3:
                contour_2d = contour_2d[0]
        z = slice_id * torch.ones(contour_2d.shape[0])[:, None]
        contour_3d = torch.cat([contour_2d, z], dim=1)
        surface_points += [contour_3d]

        # z = slice_id * torch.ones(grid.shape[0])[:, None]
        # slice_grid =torch.cat([grid, z], dim=1)
        # x_slice = x[0, slice_id]
        # xs_mesh += [slice_grid]
        # break

 
    # xs_mesh = torch.cat(xs_mesh, dim=0).float()
    surface_points = torch.cat(surface_points, dim=0).cuda().float()

    # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay1.obj', surface_points[None], None, None)
    # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay2.obj', xs_mesh[None], None, None)

    surface_points = normalize_vertices2(surface_points, shape.cuda())   

    perm = torch.randperm(len(surface_points))  
    surface_points = surface_points[perm[:np.min([len(perm), point_count])]]


    return surface_points

def contour_sdf_sampler(y, contours, i, sdf_scale_factor=2, factor=20, point_count=3000): 
     # if you increase resolution, you should increase factor as well
    D, H, W = y.shape
    x_ = torch.linspace(0, W-1, steps=(W-1)*sdf_scale_factor+1)
    y_ = torch.linspace(0, H-1, steps=(H-1)*sdf_scale_factor+1)

    grid_x, grid_y = torch.meshgrid(x_, y_, indexing='ij')
    grid = torch.cat([grid_x[:,:,None], grid_y[:,:,None]], dim=2).reshape(-1, 2).cuda() 

    # sample_grid = torch.linspace(-1, 1, factor)[:-1][None, :, None, None].cuda() # 
    # sample_grid = torch.cat([torch.zeros_like(sample_grid), sample_grid], dim=3)
    # sample_grid = torch.cat([sample_grid, sample_grid], dim=0)

    sdf = torch.ones((D, x_.shape[0], y_.shape[0]))
    sdf_idx = (grid * sdf_scale_factor).long()


    contour = contours[i]  
    for slice_id, contour_2d in contour.items():   
         
        if len(contour_2d.shape) == 3:
            contour_2d = contour_2d[0]
        # print(slice_id)
        contour_2d = contour_2d.cuda().transpose(1, 0)[:, :, None, None].float()
        contour_2d_shifted = torch.roll(contour_2d, shifts=1, dims=1)
        # contour = torch.cat([contour_2d, contour_2d_shifted], dim=2)
        # ps = F.grid_sample(contour, sample_grid, align_corners=True).reshape(2, -1).transpose(1,0)
        contour_2d = contour_2d.squeeze().transpose(1, 0)
        contour_2d_shifted = contour_2d_shifted.squeeze().transpose(1, 0)
 
        ps = contour_2d

        A = ps
        B = grid

        N1 = A.shape[0]
        N2 = B.shape[0]

        y1 = A[:, None].repeat(1, N2, 1)
        y2 = B[None].repeat(N1, 1, 1)

        diff = torch.sum((y1 - y2) ** 2, dim=2)
        dist, _ = torch.min(diff, dim=0)     
        dist = torch.sqrt(dist)

        p1 = grid
        q1 = torch.zeros_like(p1) 

        p2 = contour_2d
        q2 = contour_2d_shifted 
        intersections = doIntersect(p1, q1, p2, q2)
        intersections = torch.sum(intersections, dim=1)
        intersections = intersections % 2 == 0

        # Fix floating point error when finding inside and outside pixels
        # It is done by morphological closing (dialate and then erode)
        intersections = intersections.reshape((H-1)*sdf_scale_factor+1, (W-1)*sdf_scale_factor+1) # make it an image
        intersections = F.max_pool2d(intersections[None,None].float(), kernel_size=(3,3), stride=1, padding=(1, 1)) # dilation
        intersections = -F.max_pool2d(-intersections, kernel_size=(3,3), stride=1, padding=(1, 1))[0, 0]  # erosion
        intersections = (1-intersections).bool() # 1 - so that outside is positive (sdf is init to 1, which means non-contour slices will have 1 which is positive and is outside)
        intersections = intersections.reshape(-1) # back to original dimensions

        dist[intersections] = -dist[intersections]

        sdf[slice_id, sdf_idx[:,1], sdf_idx[:,0]] = dist.cpu() 

        
    # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay_y_hat.tif', np.float32(sdf.numpy()))

    # pip install git+https://github.com/tatsy/torchmcubes.git
    verts, faces = marching_cubes(sdf.cuda(), 0.0)  
    shape = torch.tensor(sdf.shape).flip(0)[None].float().cuda()
    # verts = normalize_vertices2(verts, shape) if normalize else verts
    verts = normalize_vertices2(verts, shape)  

     
    # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay_y_hat.tif', np.float32(sdf.numpy()))
    # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay1.obj', verts[None], faces[None], None)

    true_mesh = Meshes(verts=verts[None], faces=faces[None])
    if faces.shape[0] == 1:
        embed()
    try: 
        surface_points = sample_points_from_meshes(true_mesh, point_count)[0] 
    except RuntimeError: 
        embed()

    
    return surface_points, verts, faces


# 
# from skimage import io 
# import cv2
# patch = x.clone().cpu()[0]
# patch = 255*(patch - patch.min())/(patch.max() - patch.min())

# patch = 60*y.clone().cpu().float()
# for k, contour in contours.items():
#     for p, v in contour.items(): 
#         contour_2d, _ = v
#         img = patch[p].numpy()
#         img = np.copy(img, order='C')
#         patch[p] = torch.from_numpy(cv2.drawContours(img, [np.int64((contour_2d).cpu().numpy())], -1, (255, 255, 0), 1))  
# io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay1_{}.tif'.format(0), patch.numpy())