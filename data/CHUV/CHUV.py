from tkinter import W
from data.CHUV.register import Registration
from data.CHUV.surface_sampler import contour_sample, contour_sdf_sampler, voxel_marching_cube, voxel_surface_sampler
import numpy as np
from skimage import io
from data.data import Normalize, normalize_vertices, sample_outer_surface_in_voxel, voxel2mesh, clean_border_pixels, normalize_vertices2

import sys
from utils.metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed
from utils.registration_utils import shift_2d_replace
from utils.utils_common import crop, crop_images_and_contours, DataModes, crop_indices, blend, load_yaml, permute, flip, hist_normalize
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
         
        item = self.samples[idx]
        item.detach()
    
        x = item.image_vol 
        y = item.mask_wall
        resolution = torch.from_numpy(item.get_dicom_resolution()).cuda().float()
        contours = item.all_contours
        mode = self.mode
        config = self.cfg 

        # if mode == DataModes.TRAINING:  # if training do augmentation
        # # if True:  # if training do augmentation
        #     max_misalignment = 2
        #     shifts = torch.round(2 * (torch.rand(x.shape[0],2) - 0.5) * max_misalignment).int()
              
        #     shifted_slices = []
        #     for c, all_contours_cls_ in contours.items():  
        #         for slice_id, contour_2d in all_contours_cls_.items():  
        #             contour_2d += shifts[slice_id][None]
        #             if slice_id not in shifted_slices:  # Each slice should only be shifted once.
        #                 x[slice_id] = shift_2d_replace(data=x[slice_id], dx=shifts[slice_id][0].detach().numpy().item(), dy=shifts[slice_id][1].detach().numpy().item())
        #                 y[slice_id] = shift_2d_replace(data=y[slice_id], dx=shifts[slice_id][0].detach().numpy().item(), dy=shifts[slice_id][1].detach().numpy().item())
        #                 shifted_slices += [slice_id]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y).long().cuda()        

        fg = torch.nonzero(y>0)
        center = fg.float().mean(dim=0).long()
        center[0] = x.shape[0]//2
        center = center.cpu() 

        if mode == DataModes.TRAINING:
            # random crop during training
            shift = 2*(torch.rand(3) - 0.5)*config.shift_factor
            shift[0] = 0
            center += shift.long() 
 
        patches, contours = crop_images_and_contours([x,y], contours, config.patch_shape, tuple(center)) 
        x, y = patches

        # remove slices that don't have annotations
        # if mode == DataModes.TRAINING:  # if training do augmentation 
        if True:
            nonzero = y.cpu().sum(dim=[1,2]).nonzero()[:,0]
            clip_min = nonzero.min()
            clip_max = nonzero.max()+1 
            x[:clip_min] = 0 * x[:clip_min] 
            x[clip_max:] = 0 * x[clip_max:] 

        # hist normalization
        x = hist_normalize(x)

        # move to gpu
        x = torch.from_numpy(x).cuda().float()
  
        # augmentation done only during training
        if mode == DataModes.TRAINING:  # if training do augmentation
            if torch.rand(1)[0] > 0.5: 
                (x, y), contours = permute([x, y], contours) 
  
            if torch.rand(1)[0] > 0.5: 
                dims = random.sample([0,1,2], np.random.randint(1,4)) 
                dims = [0,1,2]
                (x, y), contours = flip([x, y], dims, contours)  

            # Brightness and contrast augmentation
            brightness = -10 + config.brightness_factor*torch.rand(1).cuda()
            contrast = 0.9 + config.contrast_factor*torch.rand(1).cuda() 
            x = torch.clamp( contrast*(x - 128) + 128 + brightness, 0, 255)


        x = (x[None] - x.mean())/x.std()  
        # x = F.interpolate(x[None], scale_factor=(1, 1/2.25, 1/2.25), mode='trilinear', align_corners=True, recompute_scale_factor=False)[0]
        # y = F.interpolate(y[None, None].float(), scale_factor=(1, 1/2.25, 1/2.25), mode='nearest', recompute_scale_factor=False)[0, 0].long() 

        x_high_res = x.clone() 
        y_high_res = y.clone()

        x = torch.cat([x, self.base_grid], dim=0) 

        surface_points_all = []  
        true_verts_all = []
        true_faces_all = []
        true_contours_all = []

 
        shape = torch.tensor(x.shape[1:]).flip(0)[None].float().cuda()
 

        for i in config.class_ids:   
            if mode == DataModes.TRAINING:
                # surface_points = voxel_surface_sampler(y, i, point_count=3000) 
                surface_points, _, _ = contour_sdf_sampler(y, contours, i, sdf_scale_factor=2, factor=20) 
            else: 
                slice_contours = contour_sample(x, y, contours, i, point_count=3000, resolution=resolution[:,:2])  
                surface_points, true_verts, true_faces = contour_sdf_sampler(y, contours, i, sdf_scale_factor=2, factor=20) 
                # true_verts_all += [(true_verts/2 + 0.5) * (shape-1) * resolution]    
                true_verts_all += [true_verts]
                true_faces_all += [true_faces]
                true_contours_all += [slice_contours]
 
            # surface_points_all += [(surface_points/2 + 0.5) * (shape-1) * resolution]    
            surface_points_all += [surface_points] 
         
        if mode == DataModes.TRAINING:
            return {   'x': x,  
                    'y_voxels': y, 
                    'x_high_res': x_high_res,
                    'y_voxels_high_res': y_high_res,  
                    'surface_points': surface_points_all, 
                    'contours': contours,
                    'unpool': [0, 1, 0, 1, 0],
                    'name':item.name,
                    'resolution': resolution
                    }
        else:
            return {   'x': x, 
                    'y_voxels': y, 
                    'x_high_res': x_high_res,
                    'y_voxels_high_res': y_high_res, 
                    'true_verts': true_verts_all, 
                    'true_faces': true_faces_all, 
                    'true_contours': true_contours_all,
                    'surface_points': surface_points_all, 
                    'contours': contours,
                    'name': item.name,
                    'resolution': resolution,
                    'unpool':[0, 1, 0, 1, 1]}
  
 