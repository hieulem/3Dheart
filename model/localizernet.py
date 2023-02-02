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

from utils.utils_common import crop_and_merge  
from utils.utils_voxel2mesh.graph_conv import adjacency_matrix, Features2Features, Feature2VertexLayer 
from utils.utils_voxel2mesh.feature_sampling import LearntNeighbourhoodSampling, NeighbourhoodSampling
from utils.utils_voxel2mesh.file_handle import read_obj 
from utils.utils_voxel2mesh.file_handle import save_to_obj

from utils.utils_voxel2mesh.unpooling import uniform_unpool, adoptive_unpool

from utils.utils_unet import UNetLayer


  
 
 
class LocalizerNet(nn.Module):
    """ LocalizerNet  """
 
    def __init__(self, config):
        super(LocalizerNet, self).__init__()

        self.config = config

        # steps = config.steps 
        steps = 5 # it's 6, manually added one more
        first_layer_channels = 16 # 8
              
        assert config.ndims ==3, Exception("Invalid nidm: {}".format(config.ndims))
        max_pool = nn.MaxPool3d([2,2,2])  
        ConvLayer = nn.Conv3d 
        ConvTransposeLayer = nn.ConvTranspose3d 
   
        '''  Down layers '''
        down_layers = [(UNetLayer(config.num_input_channels, first_layer_channels, config.ndims, kernel_size=[3,3,3], padding=[1,1,1]), max_pool)]
        for i in range(1, steps + 1):
            lyr = UNetLayer(first_layer_channels * 2**(i - 1), first_layer_channels * 2**i, config.ndims, kernel_size=[3,3,3], padding=[1,1,1])
            down_layers.append((lyr, max_pool))
        
        i = i+1
        lyr = UNetLayer(first_layer_channels * 2**(i - 1), first_layer_channels * 2**i, config.ndims, kernel_size=[1,3,3], padding=[0,1,1])
        down_layers.append((lyr, nn.MaxPool3d([1,2,2])))

        ''' Up layers '''
        up_layers = [] 
        i = steps
        upconv = ConvTransposeLayer(in_channels=first_layer_channels * 2**(i+1), out_channels=first_layer_channels * 2**i, kernel_size=[1,2,2], stride=[1,2,2])
        lyr = UNetLayer(first_layer_channels * 2**(i + 1), first_layer_channels * 2**i, config.ndims, kernel_size=[1,3,3], padding=[0,1,1])
        up_layers.append((upconv, lyr))
        for i in range(steps - 1, -1, -1):  
            upconv = ConvTransposeLayer(in_channels=first_layer_channels * 2**(i+1), out_channels=first_layer_channels * 2**i, kernel_size=[2,2,2], stride=[2,2,2])
            lyr = UNetLayer(first_layer_channels * 2**(i + 1), first_layer_channels * 2**i, config.ndims, kernel_size=[3,3,3], padding=[1,1,1])
            up_layers.append((upconv, lyr))

        ''' Final layer '''
        final_layer = ConvLayer(in_channels=first_layer_channels, out_channels=2, kernel_size=1)
 

        self.down_layers = down_layers
        self.up_layers = up_layers

        self.down = nn.Sequential(*chain(*down_layers))
        self.up = nn.Sequential(*chain(*up_layers)) 
        self.final_layer = final_layer

        # embed() 
        output_var_count = 2
        feature_channel_count = 128 # 64, total 256,512
        # self.fc_loc_laye1_size = feature_channel_count * 17 * 13
        # self.fc_loc_laye1_size = feature_channel_count * 8 * 8
        self.fc_loc_laye1_size = feature_channel_count * 5 * 4
        # self.fc_loc_laye1_size = feature_channel_count * 5 * 4

        self.localization = lambda x: x[:, :feature_channel_count]
 
        self.fc_stack_center = nn.Sequential(
                        nn.Linear(self.fc_loc_laye1_size, self.fc_loc_laye1_size // 16),        nn.ReLU(True),
                        nn.Linear(self.fc_loc_laye1_size // 16, self.fc_loc_laye1_size // 64),  nn.ReLU(True),
                        nn.Linear(self.fc_loc_laye1_size // 64, output_var_count)) 

        self.fc_stack_center[-1].weight.data.fill_(0.0)
        self.fc_stack_center[-1].bias[0].data.fill_(self.config.patch_shape_before_crop[1]//2)
        self.fc_stack_center[-1].bias[1].data.fill_(self.config.patch_shape_before_crop[2]//2)
        
        # print('just v2m +-fbq-+')

  
    def forward(self, data, iteration=0): 
        
        image = data['x']   
        # first layer
        x = self.down_layers[0][0](image)
        down_outputs = [x]

        # down layers
        for (unet_layer, max_pool) in self.down_layers[1:]: 
            x = max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x) 
 
        x_ = self.localization(x) 
        x_ = x_.reshape(x.shape[0], -1)
        stack_center = self.fc_stack_center(x_)
        
        # up layers
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers, down_outputs[-2::-1]):
            x = upconv_layer(x)
            x = crop_and_merge(down_output, x)
            x = unet_layer(x)

        seg = self.final_layer(x)  
        return seg, stack_center


    # def loss(self, data, iteration):
    def loss(self, pred, data):

        yhat_seg, yhat_centers, _, _ = pred
         
        # yhat_seg, yhat_centers = self.forward(data, iteration)   
        

        y_voxels = data['y_voxels_before_crop']
        y_voxels[y_voxels>1] = 1

        y_stack_center = data['y_stack_center_before_crop']

        CE_Loss = nn.CrossEntropyLoss()
        MSE_Loss = nn.MSELoss()

        # embed()
        ce_loss = CE_Loss(yhat_seg,  y_voxels)

        mse_loss = MSE_Loss(yhat_centers, y_stack_center.float()) 

        loss = 1 * mse_loss + 1 * ce_loss

        # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/true_voxels2.tif', np.uint8(255*data['y_dist'][0].cpu().numpy())) 
  
        log = {"loss_loc": loss.detach(), "ce_loss_loc": ce_loss.detach(), "mse_loss_loc": mse_loss.detach()}
        return loss, log


 

 

