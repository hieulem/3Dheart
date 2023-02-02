import os
from sched import scheduler
import cv2
from data.CHUV.unsupervised_register import UnsupervisedSliceRegistration
import torch
# from data.CHUV.CHUV import HeartDataset
import numpy as np

from tqdm import tqdm
from skimage import io
from scipy.ndimage import shift 
from typing import Tuple
from utils.registration_utils import check_order, shift_2d_replace, reorder_data
from utils.utils_common import load_yaml
from data.CHUV.registration_models import AnalyticalContourRegistrator, OptimizationContourRegistrator, get_loss
from IPython import embed
# from model.registration_models import AnalyticalContourRegistrator, OptimizationContourRegistrator, get_loss


class Registration:
    def __init__(self,
                 configs: dict,
                 registration_type: str, 
                 stacks: None, 
                 shift_type: str,
                 save_path: str,
                 save_slices: Tuple[bool, bool],
                 plot_contours: bool) -> None:
        """
        Args:
            configs: Dictionary containing the configuration parameters.
            registration_type: type of registration to perform (analytical, anchor_optimization, free_optimization)
            mode: mode of data (train, test, val)
            shift_type: type of shift to use to sample shifted slices (interpolation, integer)
            save_path: path to save registered slices to
            save_slices: trigger to save slices
            plot_contours: trigger to plot contours
        Analytical Solution: The analytical solution is basically aligning the slices by finding the center of each
                             slice and then shifting the center of each slice to match the center of the first slice.
        Anchor Optimization: The anchor optimization consists of an optimization procedure that finds the best shifts
                             with respect to the first slice. The loss is computed by warping each contour to the first
                             contour.
        Free Optimization: The free optimization consists of an optimization procedure that finds the best shifts with
                           respect to every slice. The loss is computed by warping each contour to every other contour.
        """
 
        self.configs = configs 
        self.stacks = stacks
        self.save_slices = save_slices
        self.plot_contours = plot_contours
        self.shift_type = shift_type
        self.registration_type = registration_type
        print(f"Doing {registration_type} registration")

    def registration(self):
        """
        Performs registration of slices with contours in dataset.
        """
        cls = 1
        print('new')

        for stack in tqdm(self.stacks):
            # Stack == patient 
            # LV = stack.all_contours 
            # outer contains the 2d/3d contours of the LV and the slices

            # all_centers_cls = stack.all_centers[cls]
 
            # centers = list(all_centers_cls.values())
            # slice_ids = list(all_centers_cls.keys())
            # num_contours = len(all_centers_cls.keys())
            # volume = stack.slices  
            # for i in range(len(centers)):
            #     centers[i] = centers[i].cuda() 

            # if self.registration_type == "unsupervised":
            #     registrator = UnsupervisedSliceRegistration(self.configs) 
            #     shifts = registrator.register(volume=volume)
            # elif self.registration_type == "anchor_optimization" or "moving_anchor_optimization":
            #     # registrator = OptimizationContourRegistrator(registration_type=self.configs.CONTOUR_REGISTRATION.type,
            #     #                                                 num_contours=num_contours,
            #     #                                                 optimizer=torch.optim.Adam,
            #     #                                                 loss_fn=get_loss(self.configs.CONTOUR_REGISTRATION.loss_fn),
            #     #                                                 lr=self.configs.CONTOUR_REGISTRATION.lr,
            #     #                                                 num_iterations=5000)
            #     # registrator.cuda()
            #     # shifts = registrator.register(contours=centers,
            #     #                                 milestones=[10, 5000],
            #     #                                 verbose=True)

            #     shifts = []
            #     for c in centers[1:]:
            #         shifts += [centers[0]-c]
            #     shifts = torch.cat(shifts,dim=0)
            # else:
            #     print('Unsupported registration') 


            # shifts_mapped = {}
            # for slice_id, shift in zip(slice_ids[1:], shifts):
            #     shifts_mapped[slice_id] = shift.cpu().detach()

            shifts_mapped = stack.shifts

            # embed()
             
            shifted_slices = []
            for c, all_contours_cls_ in stack.all_contours.items():  
                all_centers_cls_ = stack.all_centers[c]
                for slice_id, contour_2d in all_contours_cls_.items(): 
                    if slice_id in shifts_mapped.keys():
                        
                        contour_2d += shifts_mapped[slice_id]
                        all_centers_cls_[slice_id] += shifts_mapped[slice_id]
                        if slice_id not in shifted_slices:  # Each slice should only be shifted once.
                            stack.slices[slice_id].image = shift_2d_replace(data=stack.slices[slice_id].image.squeeze(),
                                                                                dx=round(shifts_mapped[slice_id][0, 0].detach().numpy().item()),
                                                                                dy=round(shifts_mapped[slice_id][0, 1].detach().numpy().item()))
                            shifted_slices += [slice_id]


            stack.compute_volumes(filled=True)
            # for c, all_contours_cls_ in stack.all_contours.items():  
            #     surface_points =[]
            #     for slice_id, contour_2d in all_contours_cls_.items():  
            #         contour_2d = contour_2d.clone() 
            #         res_z_xy_ratio = stack.get_dicom_resolution()[0,2]/stack.get_dicom_resolution()[0,1]
            #         z = res_z_xy_ratio * slice_id * torch.ones(contour_2d.shape[0])[:, None]
            #         contour_3d = torch.cat([contour_2d, z], dim=1)
            #         surface_points += [contour_3d]
            #     surface_points = torch.cat(surface_points, dim=0)
            #     save_to_obj(f'/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay4_{c}.obj', surface_points[None], None, None)

            # surface_points ={}
            # for cls, contours in stack.points3d.items():
            #     for slice_id, slice_contour in contours.items():
            #         (contour_2d, _) = slice_contour
            #         if slice_id > 0:
            #             contour_2d += shifts[slice_id-1][None]
            #         z = 4.535147392290249 * slice_id * torch.ones(contour_2d.shape[0])[:, None]
            #         contour_3d = torch.cat([contour_2d, z], dim=1) 
            #         if cls not in surface_points.keys(): 
            #             surface_points[cls] = []
            #         surface_points[cls] += [contour_3d] 
            # for k in surface_points.keys():
            #     surface_points[k] = torch.cat(surface_points[k], dim=0)
            #     save_to_obj(f'/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay5_{k}.obj', surface_points[k][None], None, None)
 

            # for shift_index, slice_id in enumerate(slice_ids[1:]): 
            #     if self.shift_type == "interpolation":
            #         stack.slices[slice_id].image = shift(input=stack.slices[slice_id].image.squeeze(),
            #                               shift=np.flip(
            #                                   np.transpose(shifts[shift_index].unsqueeze(0).detach().numpy())))
            #         for contour in stack.slices[slice_id].contours:
            #             contour.points += shifts[shift_index][None]

            #     elif self.shift_type == "integer":
            #         shifts = shifts.detach()
            #         stack.slices[slice_id].image = shift_2d_replace(data=stack.slices[slice_id].image.squeeze(),
            #                                          dx=round(shifts[shift_index][0].numpy().item()),
            #                                          dy=round(shifts[shift_index][1].numpy().item()))
            #         for contour in stack.slices[slice_id].contours:
            #             contour.points += shifts[shift_index][None].numpy()
            #     else:
            #         raise Exception("Specify correct shift type from {}.".format(["interpolation", "integer"]))
  


if __name__ == "__main__":
    configs = load_yaml("./configs/registration_config.yaml")
    R = Registration(configs=configs,
                     registration_type=configs.CONTOUR_REGISTRATION.type,
                     mode=configs.DATA.mode,
                     shift_type=configs.CONTOUR_REGISTRATION.shift_type,
                     save_path=configs.PATHS.save_path,
                     save_slices=(configs.PLOT_AND_SAVE.save_unaligned, configs.PLOT_AND_SAVE.save_aligned),
                     plot_contours=configs.PLOT_AND_SAVE.plot_contours)
    R.registration()