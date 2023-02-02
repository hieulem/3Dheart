from configs.config_CHUV import load_config
from IPython import embed 
from utils.utils_common import Modes
from data.CHUV.load_stack import load_stack, Stack, xml2contours, compute_masks, get_scar_regions_slice
from utils.utils_common import blend_cpu, blend_cpu2
import torch
import numpy as np
from utils.utils_voxel2mesh.file_handle import save_to_obj
from skimage import io

cfg = load_config(1)
save_at = '/cvlabdata2/cvlab/datasets_udaranga/outputs'

data_obj = cfg.data_obj 
dataset = data_obj.quick_load_data(cfg, 1)

data_mode = Modes.TRAINING
# data_mode = DataModes.TESTING
# embed()
for stack in dataset[data_mode].samples: 

    # ----------------------------
    # Image, LV in voxels 
    # For Paul
    # ----------------------------

    # Recompute masks with different fill 
    # True: filled region, current AC needs this
    # False: contours only, can be used for visualization and AC can also use them
    # compute_masks(stack, filled=fill)
    
    image_vol = stack.image_vol
    mask_wall = stack.mask_wall

    # save the stack (for visualization)
    image_vol = torch.from_numpy(image_vol)
    image_vol = (image_vol - image_vol.min())/(image_vol.max()-image_vol.min()) 
    overlayed_image = blend_cpu2(image_vol, torch.from_numpy(mask_wall), 4, factor=0.8)
    io.imsave(f'{save_at}/stack_{stack.name}.tif', overlayed_image)

    # ----------------------------
    # Image in voxels, LV as 3D point cloud
    # For Akbar
    # ---------------------------- 
    LV = stack.get_3d_points() 
    outer=LV[1] 
    inner=LV[2] 

    outer_points = []
    for (contour_2d, contour_3d, image) in outer:
        outer_points += [contour_3d]
    outer_points = torch.cat(outer_points, dim=0)


    inner_points = []
    for (contour_2d, contour_3d, image) in outer:
        inner_points += [contour_3d]
    inner_points = torch.cat(inner_points, dim=0)
 

    save_to_obj(f'{save_at}/inner_{stack.name}.obj', inner_points[None], [], None)  
    save_to_obj(f'{save_at}/outter_{stack.name}.obj', outer_points[None], [], None)  

    break