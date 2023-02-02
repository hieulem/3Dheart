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
from utils.utils_common import DataModes
 
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
         
    
        x = torch.from_numpy(item.image_vol).cuda().float()
        y = torch.zeros_like(x).long()  
        # y[10:20, 30:90, 30:90] = 1 
        y_refROIs = torch.zeros_like(x).long()  
        y_dist = torch.zeros_like(x).float()      
        resolution = torch.from_numpy(item.get_dicom_resolution()).cuda().float() 
        contours = {}
        mode = self.mode
        config = self.cfg 

        resolution[0,0] = 1
        resolution[0,1] = 1 
        
        x = x - x.min()
        x = torch.clamp(x/torch.quantile(x, 0.995), min=0, max=1)
 
        center = torch.tensor(x.shape)//2
        center = center.cpu() 
  

        patches, contours = crop_images_and_contours([x,y, y_dist, y_refROIs], contours, config.patch_shape, tuple(center)) 
        x, y, y_dist, y_refROIs = patches
         
        y_scar_th = y
    
        x = torch.cat([x[None], self.base_grid], dim=0) 

        y_centers_coords = {}
        for p in np.arange(32):     
            y_centers_coords[p] = torch.tensor([32, 32]).cuda().long()
 

        return {   'x': x.detach(),   
                'y_voxels': y.detach(),   
                'y_dist': y_dist.detach(),
                'y_center': y_centers_coords,
                'y_scar': y_scar_th.detach(),
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