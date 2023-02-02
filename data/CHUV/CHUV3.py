from tkinter import W
from data.CHUV.register import Registration
from data.CHUV.surface_sampler import contour_sample, contour_sdf_sampler, voxel_marching_cube, voxel_surface_sampler
import numpy as np
from skimage import io
from data.data import Normalize, normalize_vertices, sample_outer_surface_in_voxel, voxel2mesh, clean_border_pixels, normalize_vertices2

import sys
from utils.metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed
from utils.registration_utils import shift_2d_replace
from utils.utils_common import crop, crop_images_and_contours, Modes, crop_indices, blend, load_yaml, permute, flip, hist_normalize
from utils.line_intersect import doIntersect, Point

# from utils.utils_mesh import sample_outer_surface, get_extremity_landmarks, voxel2mesh, clean_border_pixels, sample_outer_surface_in_voxel, normalize_vertices 

 
# from utils import stns
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA
import pickle
import torch.nn.functional as F
from numpy.linalg import norm
import itertools as itr
import torch
from scipy import ndimage
import os 
import nibabel as nib
from IPython import embed
from os import listdir
from pydicom.filereader import dcmread
from data.CHUV.load_stack import load_stack, Stack, xml2contours, get_scar_regions_slice
from utils.utils_common import blend_cpu, blend_cpu2
import random

from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance,  mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)

import time 
from torchmcubes import marching_cubes
# from utils.utils_common import DataModes
 
class HeartDataset():

    def __init__(self, data, cfg, mode):  

        self.samples = data  

        self.cfg = cfg
        self.mode = mode

        D, H, W = cfg.patch_shape
        base_grid = torch.zeros((1, D, H, W, 3))

        w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
        h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
        d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

        base_grid[:, :, :, :, 0] = w_points
        base_grid[:, :, :, :, 1] = h_points
        base_grid[:, :, :, :, 2] = d_points 
        self.base_grid = base_grid.swapaxes(0, 4).squeeze().cuda()
         

    def __len__(self): 
        return len(self.samples)


    def __getitem__(self, idx): 

        th1 = 2
        th2 = 5
        contrast_factor = 0.2
        brightness_factor = 0.2

        item = self.samples[idx]
        item.detach()
    
        x = torch.from_numpy(item.image_vol).cuda().float()
        y = torch.from_numpy(item.mask_wall).cuda().long()   
        y_refROIs = torch.from_numpy(item.mask_scar).cuda().long()  
        y_dist = torch.from_numpy(item.mask_center).cuda().float()      
        resolution = torch.from_numpy(item.get_dicom_resolution()).cuda().float() 
        contours = item.all_contours
        mode = self.mode
        config = self.cfg 
 
        resolution[0,0] = 1
        resolution[0,1] = 1 
 
        # Make it portrait
        D, H, W = x.shape 
        if H/W < 1:
            (x, y, y_dist, y_refROIs), contours = permute([x, y, y_dist, y_refROIs], contours)        

        x = x - x.min()
        x = torch.clamp(x/torch.quantile(x, 0.995), min=0, max=1)

        # h_resolution, w_resolution, d_resolution = resolution[0] 

        # _, H, W = x.shape  
        # H = int(H * h_resolution) # 
        # W = int(W * w_resolution)  #   
        # # we resample such that 1 pixel is 1 mm in x,y and z directiions
        # base_grid = torch.zeros((1, H, W, 2))
        # w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
        # h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1) 
        # base_grid[:, :, :, 0] = w_points
        # base_grid[:, :, :, 1] = h_points 
        # grid = base_grid.cuda()
             
        # x = F.grid_sample(x[None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0]
        # y = F.grid_sample(y[None], grid, mode='nearest', padding_mode='border', align_corners=True)[0].long()
        # y_refROIs = F.grid_sample(y_refROIs[None], grid, mode='nearest', padding_mode='border', align_corners=True)[0].long()
        # y_dist = F.grid_sample(y_dist[None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0]
        # 
        
        fg = torch.nonzero(y>0)
        center = fg.float().mean(dim=0).long()
        center[0] = x.shape[0]//2
        # center = (torch.tensor(x.shape)/2).long()
        center = center.cpu() 
 
        # if mode == Modes.TRAINING:
        #     # random crop during training
        #     shift = 2*(torch.rand(3) - 0.5)*config.shift_factor
        #     shift[0] = 0
        #     center += shift.long() 


        patches, contours = crop_images_and_contours([x,y, y_dist, y_refROIs], contours, config.patch_shape, tuple(center)) 
        x, y, y_dist, y_refROIs = patches
        
        
        # remove slices that don't have annotations
        # augmentation done only during training
        if mode == Modes.TRAINING:   
            nonzero = y.cpu().sum(dim=[1,2]).nonzero()[:,0]
            clip_min = nonzero.min()
            clip_max = nonzero.max()+1 
            x[:clip_min] = 0 * x[:clip_min] 
            x[clip_max:] = 0 * x[clip_max:] 

        # augmentation done only during training
        if mode == Modes.TRAINING:  # if training do augmentation
            if torch.rand(1)[0] > 0.5: 
                dims = random.sample([0,1,2], np.random.randint(1,4))  
                (x, y, y_dist, y_refROIs), contours = flip([x, y, y_dist, y_refROIs], dims, contours)  
 
        y_myocardium = y==2 # wall
        x_scar = x.clone()
        x_scar[y_refROIs==0] = 0
        
        mu_scar = torch.sum(x_scar, dim=[1,2])/torch.sum(y_refROIs, dim=[1,2])

        std_scar = (x.clone()-mu_scar[:, None, None]) ** 2
        std_scar[y_refROIs==0] = 0
        std_scar = torch.sqrt(torch.sum(std_scar, dim=[1,2])/(torch.sum(y_refROIs, dim=[1,2])-1))

        # for i in range(32):
        #     slice_ = x_original[i]
        #     scar_ = y_scar[i]==1
        #     vals = slice_[scar_]
        #     if len(vals) > 0:
        #         print(vals.std())
 
        # y_scar_th = (x_original - mu_scar[:, None, None])/std_scar[:, None, None]

        # mu_scar = x_scar.mean()
        # std_scar = x_scar.std() 

        if self.cfg.regress_scar: 
            y_scar_th = (x - mu_scar[:, None, None])/std_scar[:, None, None]
            y_scar_th[torch.isnan(y_scar_th)] = 0.0
        else:
            y_scar_th1 = y_myocardium * (x > (mu_scar[:, None, None] + th1*std_scar[:, None, None]))
            y_scar_th2 = y_myocardium * (x > (mu_scar[:, None, None] + th2*std_scar[:, None, None]))  
            y_scar_th =  y_myocardium.long()
            y_scar_th[y_scar_th1>0] = 2 # red
            y_scar_th[y_scar_th2>0] = 3 # green
 
        # if mode == Modes.TRAINING:  # if training do augmentation
        #     # Brightness and contrast augmentation
        #     brightness = brightness_factor*(torch.rand(1).cuda()-0.5)*2
        #     contrast = 0.9 + contrast_factor*torch.rand(1).cuda() 
        #     x = torch.clamp( contrast*(x - 0.5) + 0.5 + brightness, 0, 1)
 
        x = torch.cat([x[None], self.base_grid], dim=0) 

        y_centers = torch.amax(y_dist, dim=(1,2))
        y_centers = torch.logical_and(y_dist==y_centers[:, None, None], y_dist > 0.5)
        y_centers_coords_ = torch.nonzero(y_centers)   
        y_centers_coords = {}
        for p in y_centers_coords_:     
            y_centers_coords[p[0].item()] = p[1:]


  
        return {   'x': x.detach(),   
                'y_voxels': y.detach(),   
                'y_dist': y_dist.detach(),
                'y_center': y_centers_coords,
                'y_scar': y_scar_th.detach(),
                'y_myocardium': y_myocardium.detach(),
                'mu_scar': mu_scar,
                'std_scar': std_scar,
                'contours': contours,
                'name':item.name,
                'resolution': resolution,
                'mode': mode,
                'unpool': [0, 1, 0, 1, 0]
                }
  

        # if mode == DataModes.TRAINING:  # if training do augmentation
        # # if True:  # if training do augmentation
        #     max_misalignment = 5
        #     shifts = torch.round(2 * (torch.rand(x.shape[0],2) - 0.5) * max_misalignment).int()
              
        #     shifted_slices = []
        #     for c, all_contours_cls_ in contours.items():  
        #         for slice_id, contour_2d in all_contours_cls_.items():  
        #             contour_2d += shifts[slice_id][None]
        #             if slice_id not in shifted_slices:  # Each slice should only be shifted once.
        #                 x[slice_id] = shift_2d_replace(data=x[slice_id], dx=shifts[slice_id][0].detach().numpy().item(), dy=shifts[slice_id][1].detach().numpy().item())
        #                 y[slice_id] = shift_2d_replace(data=y[slice_id], dx=shifts[slice_id][0].detach().numpy().item(), dy=shifts[slice_id][1].detach().numpy().item())
        #                 shifted_slices += [slice_id]