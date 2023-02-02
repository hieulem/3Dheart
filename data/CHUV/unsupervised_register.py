import os
from data.CHUV.registration_models import get_loss
import torch
import torch.nn as nn
import skimage.io as io

from tqdm import tqdm
from collections import defaultdict
from utils.registration_utils import shift_2d_replace
from utils.utils_common import load_yaml 
from IPython import embed
from skimage import io
import numpy as np

class UnsupervisedSliceRegistration:
    def __init__(self, configs: dict) -> None:
        """
        Performs unsupervised slice registration based on normalized cross-correlation or mutual information based on grid
        search
        
        Args:
            config: (dict) contains configuration settings for the registration
        """
        self.configs = configs
        self.normalized = self.configs.UNSUPERVISED_REGISTRATION.normalized
        self.standardized = self.configs.UNSUPERVISED_REGISTRATION.standardized
        self.padding = self.configs.UNSUPERVISED_REGISTRATION.padding
        # self.crop = self.configs.UNSUPERVISED_REGISTRATION.crop 
        # self.crop = [54, 54, 88, 88]
        # self.crop = [38, 38, 72, 72]
        self.d = 128
        self.loss_fn = get_loss(self.configs.UNSUPERVISED_REGISTRATION.loss_fn)
        self.normalized_loss = self.configs.UNSUPERVISED_REGISTRATION.normalized_loss


    def register(self, volume):
        # Number of snapshots or slices
        num_images = len(volume)

        # The first snapshot/slice is fixed
        registered_volume = [volume[0]]
        shifts = []
        for index_slice in range(1, num_images):
            fixed = registered_volume[-1].image
            cur_slice = volume[index_slice].image
 
            # import numpy as np
            

            shape = np.array(cur_slice.shape) 
            center = shape//2 
            cur_slice = cur_slice[center[0]-self.d//2:center[0]+self.d//2, center[1]-self.d//2:center[1]+self.d//2]
            # cur_slice = cur_slice[self.crop[0]:-self.crop[1], self.crop[2]:-self.crop[3]]
            cur_slice = torch.from_numpy(cur_slice).cuda()
            fixed = torch.from_numpy(fixed).cuda()
            # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay1.tif', np.uint16(cur_slice))

            

            # if self.standardized:
            #     fixed = (fixed - torch.mean(fixed)) / torch.std(fixed)
            #     cur_slice = (cur_slice - torch.mean(cur_slice)) / torch.std(cur_slice)

            # if self.normalized:
            #     fixed = (fixed - torch.min(fixed)) / (torch.max(fixed) - torch.min(fixed))
            #     cur_slice = (cur_slice - torch.min(cur_slice)) / (torch.max(cur_slice) - torch.min(cur_slice))

            # if self.crop:
            
            best_x, best_y = self.grid_search(fixed, cur_slice, index_slice)
            
            shifts += [torch.tensor([[best_x, best_y]])]

            # registered_cur = shift_2d_replace(data=volume[index_slice],
            #                                   dx=best_x,
            #                                   dy=best_y,
            #                                   constant=0.0)

            # registered_volume.append(registered_cur)

        # registered_volume = torch.stack(registered_volume, dim=0)
        shifts = torch.cat(shifts, dim=0) 
        return shifts

    def grid_search(self, fixed: torch.Tensor, cur_slice: torch.Tensor, slice_index: int) -> tuple:
        """
        Performs grid search to find the best shift for the current slice

        Args:
            fixed: fixed slice (or snapshot). It's the adjacent slice to the current slice
            cur_slice: current slice. It's adjacent slice to the fixed slice
            slice_index: index of the current slice (or snapshot)
        :return:
            best_x_y: best shift in x and y direction
        """
        cur_slice_crop_height, cur_slice_crop_width = cur_slice.shape[0], cur_slice.shape[1]

        best_measure = -float("inf")
        best_x_y = (0, 0)

        cur_slice = (cur_slice - torch.min(cur_slice)) / (torch.max(cur_slice) - torch.min(cur_slice))

        shape = np.array(fixed.shape) 
        center = shape//2 

        for x in range(-self.padding, self.padding):
            for y in range(-self.padding, self.padding):

                # Consider different shifts in x and y direction, and find the best shift

                image_region_of_interest = fixed[center[0]-self.d//2 + y:center[0]-self.d//2 + y + cur_slice_crop_height,
                                                 center[1]-self.d//2 + x:center[1]-self.d//2 + x + cur_slice_crop_width]

                image_region_of_interest = (image_region_of_interest - torch.min(image_region_of_interest)) / (torch.max(image_region_of_interest) - torch.min(image_region_of_interest))
                
                # Compute the similarity measure
                cur_measure = self.loss_fn(image_region_of_interest, cur_slice, self.normalized_loss)

                # If similarity measure is higher than the best one, update the best shift
                if cur_measure > best_measure:
                    best_measure = cur_measure
                    best_x_y = (x, y)

                
        print(f"Slice: {slice_index}, X:{best_x_y[0]}, Y:{best_x_y[1]}, Loss: {best_measure}, image_dim: {cur_slice.shape[0]}")
        return best_x_y

# if __name__ == "__main__":
#     configs = load_yaml("./configs/registration_config.yaml")
#
#     if not os.path.exists(f'{configs.PATHS.save_path}/unsupervised_registration/{configs.UNSUPERVISED_REGISTRATION.loss_fn}'):
#         os.makedirs(f'{configs.PATHS.save_path}/unsupervised_registration/{configs.UNSUPERVISED_REGISTRATION.loss_fn}')
#
#     R = UnsupervisedSliceRegistration(configs=configs)
#
#     ### Load data. This can be changed depending on how you want to load data.
#     dataset = load_data(configs.DATA.mode)
#     for stack in tqdm(dataset):
#         # Stack == patient
#         LV = stack.get_3d_points()
#         # outer contains the 2d/3d contours of the LV and the slices
#         outer = LV[1]
#
#         slices = []
#
#         for (_, _, image) in outer:
#             slices.append(torch.from_numpy(image))
#
#         registered_volume = R.register(volume=torch.stack(slices, dim=0))
#
#         if configs.PLOT_AND_SAVE.save_aligned:
#             # Save aligned slices
#             io.imsave(
#                 f'{configs.PATHS.save_path}/unsupervised_registration/{configs.UNSUPERVISED_REGISTRATION.loss_fn}/{stack.name}.tif',
#                 registered_volume.numpy())
