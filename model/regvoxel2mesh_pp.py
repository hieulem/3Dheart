from data.CHUV.surface_sampler import contour_sample, contour_sdf_sampler
from data.data import normalize_vertices2
from model.voxel2mesh_pp import Voxel2Mesh
import torch.nn as nn
import torch 
import torch.nn.functional as F 

from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)

import numpy as np
from itertools import product, combinations, chain
from scipy.spatial import ConvexHull

from IPython import embed 
import time
from utils.metrics import chamfer_directed, chamfer_symmetric 

from utils.utils_common import Modes, crop_and_merge  
from utils.utils_voxel2mesh.graph_conv import adjacency_matrix, Features2Features, Feature2VertexLayer 
from utils.utils_voxel2mesh.feature_sampling import LearntNeighbourhoodSampling, NeighbourhoodSampling
from utils.utils_voxel2mesh.file_handle import read_obj 
from utils.utils_voxel2mesh.file_handle import save_to_obj

from utils.utils_voxel2mesh.unpooling import uniform_unpool, adoptive_unpool

from utils.utils_unet import UNetLayer


  
 
 
class RegVoxel2Mesh(nn.Module):
    """ Voxel2Mesh  """
 
    def __init__(self, config):
        super(RegVoxel2Mesh, self).__init__()

        self.config = config

        steps = config.steps
        steps = 3
        first_layer_channels = 8
              
        assert config.ndims == 2 or config.ndims ==3, Exception("Invalid nidm: {}".format(config.ndims))
        self.max_pool = nn.MaxPool3d(2)  
        ConvLayer = nn.Conv3d if config.ndims == 3 else nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose3d if config.ndims == 3 else nn.ConvTranspose2d
 
        reg_layer_count = int(np.log2(self.config.patch_shape[1])) 
        reg_down_layers = []
        k = 3
        self.reg_max_pool = nn.MaxPool3d([1,2,2])  
        for i in range(reg_layer_count):
            if i < k:
                lyr = UNetLayer(first_layer_channels * 2**i, first_layer_channels * 2**(i+1), config.ndims, kernel_size=[1,3,3], padding=[0,1,1])
            else:
                lyr = UNetLayer(first_layer_channels * 2**k, first_layer_channels * 2**k, config.ndims, kernel_size=[1,3,3], padding=[0,1,1])
            
            reg_down_layers.append(lyr) 


        '''  Down layers '''
        down_layers = [UNetLayer(config.num_input_channels, first_layer_channels, config.ndims)]
        for i in range(1, steps + 1):
            lyr = UNetLayer(first_layer_channels * 2**(i - 1), first_layer_channels * 2**i, config.ndims)
            down_layers.append(lyr)

        ''' Up layers '''
        up_layers = []
        k_size = 2
        for i in range(steps - 1, -1, -1): 
            
            upconv = ConvTransposeLayer(in_channels=first_layer_channels   * 2**(i+1), out_channels=first_layer_channels * 2**i, kernel_size=k_size, stride=2)
            lyr = UNetLayer(first_layer_channels * 2**(i + 1), first_layer_channels * 2**i, config.ndims)
            up_layers.append((upconv, lyr))

        ''' Final layer '''
        final_layer = ConvLayer(in_channels=first_layer_channels * 2**k, out_channels=2, kernel_size=1)
        final_layer.weight.data.fill_(0.0)
        final_layer.bias.data.fill_(0.0)

        self.down_layers = down_layers
        self.up_layers = up_layers

        self.down = nn.Sequential(*down_layers)
        self.up = nn.Sequential(*chain(*up_layers))
        self.regdown = nn.Sequential(*reg_down_layers)
        self.final_layer = final_layer

        self.voxel2mesh = Voxel2Mesh(config) 
        checkpoint = torch.load('/cvlabdata2/cvlab/datasets_udaranga/experiments/voxel2mesh_pp/Experiment_080/trial_1/best_performance3/model.pth')
        self.voxel2mesh.load_state_dict(checkpoint['model_state_dict']) 
 
        # for param in self.voxel2mesh.parameters(): 
        #     param.requires_grad = False
        # print('just v2m +-fbq-+')

        # self.x = nn.Parameter(torch.zeros(32,2, requires_grad=True))
  
    def forward(self, data, iteration=0): 
        
        image = data['x'].clone()  
        y_voxels = data['y_voxels'].clone()  
        x_high_res = data['x_high_res'].clone()  
        y_voxels_high_res = data['y_voxels_high_res'].clone()  
        contours = data['contours']
        resolution = data['resolution'] 

        _, _, _, H, W = image.shape

        # first layer
        x = self.down_layers[0](image)
        down_outputs = [x]

        # down layers
        for unet_layer in self.down_layers[1:]: 
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x) 

        # up layers
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers, down_outputs[-2::-1]):
            x = upconv_layer(x)
            x = crop_and_merge(down_output, x)
            x = unet_layer(x)
  
        for unet_layer in self.regdown:  
            x = unet_layer(x) 
            x = self.reg_max_pool(x) 

        x = self.final_layer(x)  
        x = x.squeeze().transpose(1,0)  

        theta = torch.eye(3, device=x.device)[None]
        theta = theta.repeat([x.shape[0],1,1])
        theta[:, 0, 2] = x[:, 0]
        theta[:, 1, 2] = x[:, 1] 
 

        theta = theta[:, :2, :] 
        image = image.transpose(2,0).squeeze()
        x_high_res = x_high_res.transpose(2,0).squeeze().unsqueeze(1)
        y_voxels = y_voxels.transpose(1,0)
        y_voxels_high_res = y_voxels_high_res.transpose(1,0)
         
        grid = F.affine_grid(theta, image.shape, align_corners=True) 
        image = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        x_high_res = F.grid_sample(x_high_res, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        y_voxels = F.grid_sample(y_voxels.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()
        y_voxels_high_res = F.grid_sample(y_voxels_high_res.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()
        

        data['x'] = image[None].transpose(2, 1) 
        data['x_high_res'] = x_high_res[None].transpose(2, 1) 
        data['y_voxels'] = y_voxels.transpose(1, 0)
        data['y_voxels_high_res'] = y_voxels_high_res.transpose(1, 0)
        
        shape = torch.tensor([[H, W]]).cuda() 
        contours_ = {}
        for class_id, contour in contours.items():
            contours__ = {}
            for slice_id, contour_2d in contour.items():   
                contours__[slice_id] = contour_2d[0].cuda() - shape * x[slice_id][None]/2
            contours_[class_id] = contours__
 
        surface_points_all = []  
        true_verts_all = []
        true_faces_all = []
        true_contours_all = [] 
        for i in self.config.class_ids:   
            if self.training:  
                contour_3d = []
                for slice_id, contour_2d in contours_[i].items(): 
                    z = slice_id * torch.ones(contour_2d.shape[0])[:, None].cuda()
                    contour_3d_ = torch.cat([contour_2d, z], dim=1)
                    contour_3d += [contour_3d_]

                contour_3d = torch.cat(contour_3d, dim=0)
                
                shape = torch.tensor(data['x'][0, 0].shape).flip(0)[None].float().cuda()
                surface_points = normalize_vertices2(contour_3d, shape)   
                surface_points = surface_points[None]
 
            else: 
                slice_contours = contour_sample(data['x'][0], data['y_voxels'][0], contours_, i, point_count=3000, resolution=resolution[:,:2])  
                surface_points, true_verts, true_faces = contour_sdf_sampler(data['y_voxels'][0], contours_, i, sdf_scale_factor=2, factor=20) 
                true_verts_all += [true_verts[None]]
                true_faces_all += [true_faces[None]] 
 
                slice_contours_ = {}
                for k, v in slice_contours.items():
                    slice_contours_[k] = v[None].detach().cpu()
                true_contours_all += [slice_contours_]
 
            surface_points_all += [surface_points]  

        if self.training:
            data['surface_points'] = surface_points_all
        else:
            data['true_verts'] = true_verts_all
            data['true_faces'] = true_faces_all
            data['true_contours'] = true_contours_all
            data['surface_points'] = surface_points_all
 
        pred, _ = self.voxel2mesh(data)
 
        return pred, data


    def loss(self, data, iteration):

         
        pred, data = self.forward(data, iteration)   
         
        # CE_Loss = nn.CrossEntropyLoss() 
        # ce_loss = CE_Loss(pred[0][-1][3], data['y_voxels'].long())


        chamfer_loss = torch.tensor(0).float().cuda()
        edge_loss = torch.tensor(0).float().cuda()
        laplacian_loss = torch.tensor(0).float().cuda()
        normal_consistency_loss = torch.tensor(0).float().cuda()  

        for c in range(self.config.num_classes-1): 
            target = data['surface_points'][c].cuda()  

            vertices, faces, _, _, _ = pred[c][-1]
            pred_mesh = Meshes(verts=list(vertices), faces=list(faces))
            pred_points = sample_points_from_meshes(pred_mesh, 3000)

            # chamfer_loss +=  chamfer_directed(target, pred_points) 
            chamfer_loss +=  chamfer_distance(pred_points.float(), target.float())[0]
            # chamfer_loss +=  chamfer_symmetric(target, pred_points) 
            laplacian_loss +=   mesh_laplacian_smoothing(pred_mesh, method="uniform")
            normal_consistency_loss += mesh_normal_consistency(pred_mesh) 
            edge_loss += mesh_edge_loss(pred_mesh) 
            
        # print('-')
        # loss = 1 * chamfer_loss + 1 * ce_loss  
        loss = 1 * chamfer_loss + 0.1 * laplacian_loss + 1 * edge_loss + 0.1 * normal_consistency_loss
        # loss = 1 * chamfer_loss + 1 * ce_loss + 0.001 * laplacian_loss + 0.001 * edge_loss + 0.001 * normal_consistency_loss

        log = {"loss": loss.detach(),
               "chamfer_loss": chamfer_loss.detach(),  
               "normal_consistency_loss": normal_consistency_loss.detach(),
               "edge_loss": edge_loss.detach(),
               "laplacian_loss": laplacian_loss.detach()}
        return loss, log


 

 

