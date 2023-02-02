from data.CHUV.surface_sampler import contour_sample, contour_sdf_sampler
from data.data import normalize_vertices2
from model.localizernet import LocalizerNet
from model.registernet import RegisterNet
from model.scarnet import ScarNet2D
from model.voxel2mesh_pp import Voxel2Mesh
import torch.nn as nn
import torch 
import torch.nn.functional as F 
from collections import ChainMap

from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)

import numpy as np
from itertools import product, combinations, chain
from scipy.spatial import ConvexHull

from IPython import embed 
import time
from utils.metrics import chamfer_directed, chamfer_symmetric
from utils.rasterize.rasterize2 import rasterize_vol
from utils.registration_utils import shift_2d_replace 

from utils.utils_common import Modes, crop_and_merge, crop_images_and_contours  
from utils.utils_voxel2mesh.graph_conv import adjacency_matrix, Features2Features, Feature2VertexLayer 
from utils.utils_voxel2mesh.feature_sampling import LearntNeighbourhoodSampling, NeighbourhoodSampling
from utils.utils_voxel2mesh.file_handle import read_obj 
from utils.utils_voxel2mesh.file_handle import save_to_obj

from utils.utils_voxel2mesh.unpooling import uniform_unpool, adoptive_unpool

from utils.utils_unet import UNetLayer
 
class HEARTS(nn.Module):
    """ Voxel2Mesh  """
 
    def __init__(self, config):
        super(HEARTS, self).__init__()
        self.config = config
        self.register = config.register_slices
        self.localizernet = LocalizerNet(config)
        self.registernet = RegisterNet(config)
        # checkpoint = torch.load('/cvlabdata2/cvlab/datasets_udaranga/experiments/voxel2mesh_pp/Experiment_153/trial_2/best_performance3/model.pth')
        # self.localizernet.load_state_dict(checkpoint['model_state_dict']) 
 
        self.voxel2mesh = Voxel2Mesh(config) 
        # checkpoint = torch.load('/cvlabdata2/cvlab/datasets_udaranga/experiments/voxel2mesh_pp/Experiment_080/trial_1/best_performance3/model.pth')
        # self.voxel2mesh.load_state_dict(checkpoint['model_state_dict']) 
  
        self.scarnet = ScarNet2D(config)
        # checkpoint = torch.load('/cvlabdata2/cvlab/datasets_udaranga/experiments/voxel2mesh_pp/Experiment_170/trial_1/best_performance3/model.pth')
        # scarnet_ = {}
        # for k, v in checkpoint['model_state_dict'].items():
        #     if 'scar' in k:
        #         scarnet_[k[8:]] = v 
        # self.scarnet.load_state_dict(scarnet_) 
 
        # for param in self.localizernet.parameters(): 
        #     param.requires_grad = False

        # for param in self.voxel2mesh.parameters(): 
        #     param.requires_grad = False
        print('frozen localizer and v2m 253')

        # self.x = nn.Parameter(torch.zeros(32,2, requires_grad=True))
   
    def register_slices(self, data, yhat_seg_, yhat_centers_, mode):
 
        # Register slices
        yhat_seg = torch.argmax(yhat_seg_, dim=1) 
        lv_slices = torch.sum(yhat_seg[0], dim=(1,2))

        yhat_centers = {}
        for k, v in enumerate(yhat_centers_[0]): 
            if lv_slices[k] > 100: 
                yhat_centers[k] = v[None].detach() 

        shifts_mapped={} 
        slice_ids = list(yhat_centers.keys())
        for slice_id in slice_ids[1:]:
            shifts_mapped[slice_id] = yhat_centers[slice_ids[0]]-yhat_centers[slice_id]

        if mode is Modes.DEPLOY:
            x = data['x'].detach().clone() 
            for slice_id in shifts_mapped.keys():
                yhat_centers[slice_id] += shifts_mapped[slice_id] 
                x[0, 0, slice_id] = shift_2d_replace(data=x[0, 0, slice_id],
                                                    dx=round(shifts_mapped[slice_id][0, 1].detach().item()),
                                                    dy=round(shifts_mapped[slice_id][0, 0].detach().item())) 
            data['x'] = x  
        else: 
            x = data['x'].detach().clone() 
            y = data['y_voxels']
            y_dist = data['y_dist']
            y_scar = data['y_scar'] 
            y_myocardium = data['y_myocardium'] 
            contours = data['contours']
      
            for slice_id in shifts_mapped.keys():
                yhat_centers[slice_id] += shifts_mapped[slice_id] 
                x[0, 0, slice_id] = shift_2d_replace(data=x[0, 0, slice_id],
                                                    dx=round(shifts_mapped[slice_id][0, 1].detach().item()),
                                                    dy=round(shifts_mapped[slice_id][0, 0].detach().item()))       

                yhat_seg_[0,0, slice_id] = shift_2d_replace(data=yhat_seg_[0,0, slice_id],
                                                    dx=round(shifts_mapped[slice_id][0, 1].detach().item()),
                                                    dy=round(shifts_mapped[slice_id][0, 0].detach().item()))

                yhat_seg_[0,1, slice_id] = shift_2d_replace(data=yhat_seg_[0,1, slice_id],
                                                    dx=round(shifts_mapped[slice_id][0, 1].detach().item()),
                                                    dy=round(shifts_mapped[slice_id][0, 0].detach().item()))        
            
            shifted_slices = []
            for c, all_contours_cls_ in contours.items():   
                for slice_id, contour_2d in all_contours_cls_.items(): 
                    if slice_id in shifts_mapped.keys():   
                        contour_2d += torch.flip(shifts_mapped[slice_id].cpu()[None] , dims=[2]) 

                        if slice_id not in shifted_slices:  # Each slice should only be shifted once.   
                            y[0, slice_id] = shift_2d_replace(data=y[0, slice_id],
                                                                dx=round(shifts_mapped[slice_id][0, 1].detach().item()),
                                                                dy=round(shifts_mapped[slice_id][0, 0].detach().item()))
    
                            y_scar[0, slice_id] = shift_2d_replace(data=y_scar[0, slice_id],
                                                                dx=round(shifts_mapped[slice_id][0, 1].detach().item()),
                                                                dy=round(shifts_mapped[slice_id][0, 0].detach().item()))
    
                            y_myocardium[0, slice_id] = shift_2d_replace(data=y_myocardium[0, slice_id],
                                                                dx=round(shifts_mapped[slice_id][0, 1].detach().item()),
                                                                dy=round(shifts_mapped[slice_id][0, 0].detach().item()))
    
                            y_dist[0, slice_id] = shift_2d_replace(data=y_dist[0, slice_id],
                                                                dx=round(shifts_mapped[slice_id][0, 1].detach().item()),
                                                                dy=round(shifts_mapped[slice_id][0, 0].detach().item())) 
                            shifted_slices += [slice_id]
            
            data['x'] = x  
            data['y_voxels'] = y
            data['y_scar'] = y_scar
            data['y_myocardium'] = y_myocardium 
            data['contours'] = contours 

        return data

    def extract_surface_data(self, data, mode):
        surface_points_all = []  
        true_verts_all = []
        true_faces_all = []
        true_contours_all = []
        resolution = data['resolution']
        contours = data['contours']
        x = data['x']
        y = data['y_voxels']
 
        if y.sum()>0: # i.e. there is annotation 
            for i in self.config.class_ids:   
                if mode == Modes.TRAINING: 
                    surface_points, _, _ = contour_sdf_sampler(y[0], contours, i, sdf_scale_factor=2, factor=20) 
                else: 
                    slice_contours = contour_sample(x[0], y[0], contours, i, point_count=3000, resolution=resolution[0, :,:2])  
                    surface_points, true_verts, true_faces = contour_sdf_sampler(y[0], contours, i, sdf_scale_factor=2, factor=20) 
                    
                    true_verts_all += [true_verts[None]]
                    true_faces_all += [true_faces[None]]
                    true_contours_all += [slice_contours]
        
                surface_points_all += [surface_points[None]] 
 
        data['surface_points'] = surface_points_all 
        data['true_verts'] = true_verts_all
        data['true_faces'] = true_faces_all
        data['true_contours'] = true_contours_all
        return data
    
    def prepare_scarenet_input(self, data, pred2):
        x = data['x']
        verts = torch.flip(pred2[0][-1][0].detach(), [2])  
        mesh = Meshes(verts=verts.detach(), faces=pred2[0][-1][1].detach().long()) 
        pred_voxels_rasterized_ = rasterize_vol(mesh, x.shape[2:]) 

        verts = torch.flip(pred2[1][-1][0].detach(), [2])  
        mesh = Meshes(verts=verts.detach(), faces=pred2[1][-1][1].detach().long()) 
        pred_voxels_rasterized = rasterize_vol(mesh, x.shape[2:]) 
        pred_voxels_rasterized[pred_voxels_rasterized_>0] = 0
        pred_voxels_rasterized = pred_voxels_rasterized.float()[None, None]
 
         
        # scarnet_input = torch.cat([pred_voxels_rasterized * x[:,0].unsqueeze(1), pred_voxels_rasterized], dim=1)
        scarnet_input = torch.cat([x[:,0].unsqueeze(1), 0 * pred_voxels_rasterized], dim=1)
        # scarnet_input = x[:,0].unsqueeze(1)
        return scarnet_input

    def forward(self, data, mode=Modes.TRAINING, center=None): 
        #  
        # # Model 1: Localize centers   
        yhat_loc_seg__, yhat_loc_centers__ = self.localizernet(data)

        yhat_loc_seg_ = yhat_loc_seg__.detach().clone()
        yhat_loc_centers_ = yhat_loc_centers__.detach().clone()  
        pred0 = (yhat_loc_seg__, yhat_loc_centers__, yhat_loc_seg_, yhat_loc_centers_)

        if mode is Modes.DEPLOY:
            x = data['x'].detach().clone() 

            if center is None:
                center = (torch.tensor(x.shape)/2).long().cpu()   
                center[3] = yhat_loc_centers_[0, 0].cpu()
                center[4] = yhat_loc_centers_[0, 1].cpu() 

            patches, _ = crop_images_and_contours([x], {}, (x.shape[0],x.shape[1])+self.config.patch_shape,  center) 
            x = patches[0]
            data['x'] = x.detach().clone()  
        else: 
            x = data['x'].detach().clone() 
            y = data['y_voxels_before_crop']
            y_dist = data['y_dist']
            y_scar = data['y_scar'] 
            y_myocardium = data['y_myocardium'] 
            contours = data['contours']
 
            # during deployment it has to be replaced with loc output
            fg = torch.nonzero(y>0)
            center = fg.float().mean(dim=0).long()
            center[1] = x.shape[2]//2
            center = center.cpu() 
            if mode == Modes.TRAINING: 
                # random crop during training
                shift = 2*(torch.rand(4) - 0.5)*self.config.shift_factor
                shift[0] = 0
                shift[1] = 0
                center += shift.long() 

            center_x = tuple([center[0],torch.tensor(2), center[1], center[2], center[3]])


            # embed()
            # center_x = (torch.tensor(x.shape)/2).long().cpu()   
            # center_x[3] = yhat_loc_centers_[0, 0].cpu()
            # center_x[4] = yhat_loc_centers_[0, 1].cpu()

            # center = (torch.tensor(y.shape)/2).long().cpu() 
            # center[2] = yhat_loc_centers_[0, 0].cpu()
            # center[3] = yhat_loc_centers_[0, 1].cpu()  
            patches, _ = crop_images_and_contours([x], {}, (x.shape[0],x.shape[1])+self.config.patch_shape,  center_x) 
            x = patches[0]
            data['x'] = x.detach().clone()  


            patches, contours = crop_images_and_contours([y, y_dist, y_scar, y_myocardium], contours, (x.shape[0],)+self.config.patch_shape, tuple(center)) 
            y, y_dist, y_scar, y_myocardium = patches

            data['y_voxels'] = y.detach().clone() 
            data['y_dist'] = y_dist.detach().clone() 
            data['y_scar'] = y_scar.detach().clone() 
            data['y_myocardium'] = y_myocardium.detach().clone() 
            data['contours'] = contours    
    

            y_centers = torch.amax(y_dist, dim=(2,3))
            y_centers = torch.logical_and(y_dist==y_centers[:, :, None, None], y_dist > 0.5)
            y_centers_coords_ = torch.nonzero(y_centers)[:,1:]  
            y_centers_coords = {}
            for p in y_centers_coords_:    
                y_centers_coords[p[0].item()] = p[1:][None].detach().clone()
            data['y_center'] = y_centers_coords    

        # ------------------------------
        yhat_reg_seg__, yhat_reg_centers__ = self.registernet(data)


        yhat_reg_seg_ = yhat_reg_seg__.detach().clone()
        yhat_reg_centers_ = yhat_reg_centers__.detach().clone() 

        # Register slices
        pred1 = (yhat_reg_seg__, yhat_reg_centers__, yhat_reg_seg_, yhat_reg_centers_) 

        data = self.register_slices(data, yhat_reg_seg_, yhat_reg_centers_, mode)    

        if mode is not Modes.DEPLOY:        
            data = self.extract_surface_data(data, mode)


        # # Model 2: Voxel2Mesh - segmentation
        pred2, _ = self.voxel2mesh(data)
     
        # # Model 3: Scar segmentation @Gaspard: you don't need pred3
        scarnet_input = self.prepare_scarenet_input(data, pred2)  

        if mode is Modes.TRAINING:
            pred3 = self.scarnet(scarnet_input.detach())
        else: 
            pred3 = self.scarnet(scarnet_input)
            for k in range(1, 4):
                input_ = torch.rot90(scarnet_input, k, [3, 4])   
                pred3 += torch.rot90(self.scarnet(input_), -k, [3, 4]) 
            for k in range(4):
                input_ = torch.flip(scarnet_input, dims=[3])
                input_ = torch.rot90(input_, k, [3, 4])
                pred3 += torch.flip(torch.rot90(self.scarnet(input_), -k, [3, 4]), dims=[3]) 
            pred3 = pred3/8

        pred = {}
        pred['localize'] = pred0
        pred['register'] = pred1
        pred['voxel2mesh'] = pred2
        pred['scarnet'] = pred3

        return pred, data


    def loss(self, pred, data):
        
        loss0, log0 = self.localizernet.loss(pred['localize'], data)
        loss1, log1 = self.registernet.loss(pred['register'], data)
        loss2, log2 = self.voxel2mesh.loss(pred['voxel2mesh'], data)
        loss3, log3 = self.scarnet.loss(pred['scarnet'], data)
        loss =  loss0 + loss1 + loss2 + loss3 
        log = dict(ChainMap(log0, log1, log2, log3))  
        return loss, log



 

 

