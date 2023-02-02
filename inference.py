 
import os
GPU_index = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index

  
import logging
import torch
import numpy as np
from train import Trainer
from evaluate import Evaluator  

from shutil import copytree, ignore_patterns
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils_common import Modes, crop, crop_images_and_contours, hist_normalize, permute
import wandb
from IPython import embed 
from torchmcubes import marching_cubes
import trimesh
import time

from utils.utils_common import mkdir
from utils.utils_voxel2mesh.file_handle import save_to_obj  

# from configs.config_ACDC import load_config
from configs.config_CHUV import load_config
# from configs.config_Liver import load_config

# from model.voxel2mesh_pp import Voxel2Mesh as network
# from model.hearts import HEARTS as network
from model.hearts2 import HEARTS as network
# from model.unet import UNet as network

# from data.CHUV.heart import Heart
from data.data import voxel2mesh, clean_border_pixels, normalize_vertices2
import cv2 as cv 
from data.CHUV.load_stack import load_stack, Stack, xml2contours 


logger = logging.getLogger(__name__)

 
def init(): 
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True  # speeds up the computation 
 

def predict(data, model_path, device):
   
    # Initialize
    cfg = load_config() 
    init()
  
    print("Create network")
    hearts = network(cfg)
    hearts.to(device)
  
    print("Loading pretrained network") 
    checkpoint = torch.load(model_path)
    hearts.load_state_dict(checkpoint['model_state_dict']) 
    
    pred, _ = hearts(data, mode=Modes.DEPLOY)  
    pred_localizer = pred['localize']
    pred_register = pred['register']
    pred_voxel2mesh = pred['voxel2mesh']
    pred_scarnet = pred['scarnet']

    # Stack center
    _, stack_center, _, _ = pred_localizer
    
    # Slice Centers
    _, slice_centers, _, _ = pred_register 

    # Voxel2Mesh output
    pred_lv_meshes = []  
    pred_scar_meshes =[]
    resolution = data['resolution'].cpu()  

    shape = torch.tensor(data['x'].shape)  
    shape = shape[2:].flip([0])[None, None] # flip() because we flip the ijk values in data laoder 

    for c in range(cfg.num_classes-1):    
        pred_vertices = pred_voxel2mesh[c][-1][0].detach().data.cpu()
        pred_faces = pred_voxel2mesh[c][-1][1].detach().data.cpu()   

        # true_vertices = data['vertices_mc'][c].data.cpu()
        # true_faces = data['faces_mc'][c].data.cpu() 
        
        pred_lv_meshes += [{'vertices': (pred_vertices/2 + 0.5) * (shape-1) * resolution, 'faces':pred_faces}] 
        # true_meshes += [{'vertices': true_vertices * scale_factor_from_res * scale_factor_from_coord_normalization, 'faces':true_faces, 'normals':None}] 

    # ScarNet Output
    yhat_scars = torch.argmax(pred_scarnet, dim=1)    
 
    for sc in [2, 3]: 
        yhat_scars_ = (yhat_scars==sc).float()[0] 
        scar_verts, scar_faces = marching_cubes(yhat_scars_, 0.5)  
        # scar_verts, scar_faces = marching_cubes(y_scar_z, 2.0)  
        scar_verts = scar_verts.cpu() * resolution[0]
        scar_faces = scar_faces.cpu() 

        mesh_ = trimesh.Trimesh(scar_verts, scar_faces)
        # mesh_ = trimesh.smoothing.filter_laplacian(mesh_, lamb=0.0) 
        scar_verts = torch.from_numpy(mesh_.vertices)
        scar_faces = torch.from_numpy(mesh_.faces) 
        pred_scar_meshes += [{'vertices': scar_verts[None], 'faces':scar_faces[None]}] 
 
    return stack_center, slice_centers, pred_lv_meshes, pred_scar_meshes, yhat_scars

def get_input(root, name, device):
    scale_factor = 1  
    # patch_shape = (32, 128, 128) 
    patch_shape_before_crop = (32, 320, 256) 
    D, H, W = patch_shape_before_crop
    base_grid = torch.zeros((1, D, H, W, 3))

    w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
    h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
    d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

    base_grid[:, :, :, :, 0] = w_points
    base_grid[:, :, :, :, 1] = h_points
    base_grid[:, :, :, :, 2] = d_points 
    base_grid = base_grid.swapaxes(0, 4).squeeze().to(device)



    path = f'{root}/{name}' 
    slices = load_stack(path, scale_factor=scale_factor) 
    # stack = Stack(slices, sample, type=type, fill=True)  

    # Since we do a crop such that the heart is more or less at the center, we need to load the contours as well.
    # TODO: another network for localization
    if len(slices) == 0: 
        print('empty stack')
    else:
        loaded = xml2contours(path, slices)
        if loaded:
            stack = Stack(slices, name, type=type, fill=True)   
            stack.compute_volumes(filled=True)
        else:
            print('error loading contours')

       
        x = torch.from_numpy(stack.image_vol).to(device)  
        resolution = torch.from_numpy(stack.get_dicom_resolution()).to(device).float() 
        resolution[0,0] = 1 # stack is resampled to have 1mm resolution in x,y directions
        resolution[0,1] = 1  

        # Make it portrait <<<<<<< @Antoine: Discuss this with me
        D, H, W = x.shape  
        if H/W < 1:
            x_, _ = permute([x], {})     
            x = x_[0]   

        x = x - x.min()
        x = torch.clamp(x/torch.quantile(x, 0.995), min=0, max=1)
 
        center = (torch.tensor(x.shape)/2).long()
        center = center.cpu() 
  
        patches, contours = crop_images_and_contours([x], {}, patch_shape_before_crop, tuple(center)) 
        x = patches[0] 
 
        x = torch.cat([x[None], base_grid], dim=0)[None]
        
    return {   'x': x,   
                'name': stack.name,
                'resolution': resolution,
                'unpool':[[0], [1], [0], [1], [0]]} 

def main():

    # ----------
    # UI Inputs from text boxes
    # ----------
    model_path = '/cvlabdata2/cvlab/datasets_udaranga/experiments/voxel2mesh_pp/Experiment_232/trial_6/best_performance3/model.pth'
    root = '/cvlabsrc1/cvlab/datasets_udaranga/datasets/3d/CHUV/dataset/1-FB-ICM-Learn'
    sample = 'FB_CMR examination_035' 
    device = "cuda"
    start = time.time()
    # ----------

    # Two options: Either run this funciton on local machine in python OR zip send 'sample' to server and run this function on server
    input = get_input(root, sample, device)
    input_loaded = time.time()

    # Run on the server
    stack_center, slice_centers, pred_lv_meshes, pred_scar_meshes, yhat_scars = predict(input, model_path, device)
    end_of_compute = time.time()

    print(f'{input_loaded-start} | {end_of_compute-input_loaded}')

    # Save to disk at the moment. Instead this should be visualized in MITK
    # At the moment the mesh is in world coordinates. Needs to be converted to slice coordinates. 
    save_path = '/cvlabdata2/cvlab/datasets_udaranga/outputs'
    for p, pred_mesh in enumerate(pred_lv_meshes):
        save_to_obj(save_path + '/pred_lv_part_' + str(p) + '.obj', pred_mesh['vertices'], pred_mesh['faces'], None)

    for p, pred_mesh in enumerate(pred_scar_meshes):
        save_to_obj(save_path + '/pred_scar_part_' + str(p) + '.obj', pred_mesh['vertices'], pred_mesh['faces'], None)
        

if __name__ == "__main__": 
    main()



# model_path = '/cvlabdata2/cvlab/datasets_udaranga/experiments/voxel2mesh_pp/Experiment_035/trial_1/best_performance3/model.pth'

# cfg = load_config() 
# print("Load pre-processed data") 
# data_obj = cfg.data_obj 
# data = data_obj.quick_load_data(cfg)
# item = data[DataModes.TESTING].samples[0]

# input = get_input(item)

# pred_meshes, true_meshes = predict(input, model_path) 
# save_path = '/cvlabdata2/cvlab/datasets_udaranga/outputs'
# for p, (true_mesh, pred_mesh) in enumerate(zip(true_meshes, pred_meshes)):
#     save_to_obj(save_path + '/true_part_' + str(p) + '.obj', true_mesh['vertices'][None], true_mesh['faces'][None], None)
#     save_to_obj(save_path + '/pred_part_' + str(p) + '.obj', pred_mesh['vertices'], pred_mesh['faces'], None)
