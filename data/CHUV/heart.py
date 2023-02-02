from tkinter import W
from data.CHUV.CHUV4 import HeartDataset
# from data.CHUV.CHUV3 import HeartDataset
from data.CHUV.register import Registration 
import numpy as np

from utils.metrics import jaccard_index, chamfer_weighted_symmetric 
from utils.utils_common import Modes, crop_images_and_contours, load_yaml 


# from utils.utils_mesh import sample_outer_surface, get_extremity_landmarks, voxel2mesh, clean_border_pixels, sample_outer_surface_in_voxel, normalize_vertices 

 
# from utils import stns
from torch.utils.data import Dataset 
import pickle 
import torch 
import os 
import nibabel as nib
from IPython import embed 
from data.CHUV.load_stack import load_snapshots, load_stack, Stack, xml2contours
  

from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
     
class Heart():

    def pick_surface_points(self, y_outer, point_count):
        idxs = torch.nonzero(y_outer) 
        perm = torch.randperm(len(idxs))

        y_outer = y_outer * 0  
        idxs = idxs[perm[:point_count]]
        y_outer[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
        return y_outer

    def quick_load_data(self, cfg, trial_id=None, scale_factor=1):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        # down_sample_shape = cfg.patch_shape

        data_root = cfg.dataset_path
        data = {}
        r = 0.8
        for i, datamode in enumerate([Modes.TRAINING, Modes.TESTING]):
            path = f'{data_root}/pre_computed_data_{datamode}_scale_factor_{scale_factor}_r_{r}_v11.pickle'
            print(path)
            with open(path, 'rb') as handle:
                samples = pickle.load(handle) 
                data[datamode] = HeartDataset(samples, cfg, datamode) 

        return data

    def pre_process_snapshot_dataset(self, cfg):
        '''         
            ----------------------------------------------
            Dataset Overview
            ----------------------------------------------
                Patient count: 1 (at the moment)
            ----------------------------------------------
             
        '''   
        scale_factor = 1
        data_root = cfg.dataset_path   
 
        path_SnapShots = f'{data_root}/1.Test_case_badMoco_Sept22_2022'
        samples_SnapShots = [dir for dir in os.listdir(path_SnapShots)] 

        samples = []

        for sample in samples_SnapShots:
            samples += [(path_SnapShots, sample, 'SnapShots')]        



        stacks = []  
        overlayed_images = []
        fill = True

        c = 0 
        p = 0
        inputs = []
        labels = []

        yys = []
        xxs = [] 
        # samples = [samples[37]]
        for k, (root, sample, type) in enumerate(samples): 
            path = f'{root}/{sample}' 
            slices_loaded = True
            contours_loaded = True   

            if os.path.isdir(path):
                print(path)   
                slices = load_snapshots(path, scale_factor=scale_factor) 
                 
                if len(slices) == 0: 
                    slices_loaded = False 
                else: 
                    
                    loaded = xml2contours(path, slices)
                    if loaded:
                        stack = Stack(slices, sample, type=type, fill=True) 
                        stack.compute_volumes(filled=True)
                         
 
 
                        dicom_resolution = stack.get_dicom_resolution()
                        gttool_resolution = stack.get_gttool_resolution()
                        if dicom_resolution.shape[0] == 1 and gttool_resolution.shape[0] == 1:
                            stacks += [stack] 
                            p += 1  
                        else:
                            print('incosistant resolution')
                            print(dicom_resolution)
                            print('-')
                            print(gttool_resolution)
 
 
                    else: 
                        contours_loaded = False  

                
                if not slices_loaded or not contours_loaded:
                    # print(f'{k} {slices_loaded} {contours_loaded} {path}  --------------')
                    c += 1
                # else: 
                    # print(f'{k} {slices_loaded} {contours_loaded} {path}')
  

 
        print('\nSaving pre-processed data to disk')
        # exit()
        np.random.seed(0)
        perm = np.random.permutation(len(stacks)) 
        
        r = 0.8
        counts = [perm[:np.floor(len(stacks)*r).astype('int')], perm[np.floor(len(stacks)*r).astype('int'):]]
        # counts = [perm, perm]
    
        for i, datamode in enumerate([Modes.TRAINING, Modes.TESTING]):

            samples = [] 

            for j in counts[i]: 
                print('.',end='', flush=True)
                # x = inputs[j]
                # y = labels[j]
                
                # print(stacks[j].slices[0].image.shape) 
                samples.append(stacks[j])  
            with open(data_root + f'/pre_computed_data_{datamode}_scale_factor_{scale_factor}_r_{r}_v11.pickle', 'wb') as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
        
        print('\nPre-processing complete!')  


    def pre_process_dataset(self, cfg):
        '''         
            ----------------------------------------------
            Dataset Overview
            ----------------------------------------------
                ICM: 62 samples (53 loaded)
                NICM: 19 samples (17 loaded)
            ----------------------------------------------
            
            ----------------------------------------------
            Not loaded 
            ---------------------------------------------- 
                misaligned contours: In ICM: {072}, in NICM: {126}
                missing 51: In ICM {062,023,046,022}, in NICM: {071}
                multiple 51 stacks: 133_PSIR,044,132,133_Localisateur and Long_axis  
            ----------------------------------------------
            
            ----------------------------------------------
            Loaded, but contour assigned to slice based on slice index instead of z coordinate. 
            This is accurate most of the time, but can fail sometimes.
            ----------------------------------------------
                ICM: {32,092,073,072,017,053,004_Localisator,096,031,063,006,001,049}
                NICM: {120,118,127,117,126}
            ----------------------------------------------
        '''  


        scale_factor = 1
        data_root = cfg.dataset_path  
        path_ICM = f'{data_root}/1-FB-ICM-Learn' 
        path_NICM = f'{data_root}/1-FB-NICM-Learn'


        samples_ICM = [dir for dir in os.listdir(path_ICM)]
        samples_NICM = [dir for dir in os.listdir(path_NICM)]

        samples = []

        for sample in samples_ICM:
            samples += [(path_ICM, sample, 'ICM')]

        for sample in samples_NICM:
            samples += [(path_NICM, sample, 'NICM')]
 
        # path_SnapShots = f'{data_root}/1.Test_case_badMoco_Sept22_2022'
        # samples_SnapShots = [dir for dir in os.listdir(path_SnapShots)] 

        # samples = []

        # for sample in samples_SnapShots:
        #     samples += [(path_SnapShots, sample, 'SnapShots')]        



        stacks = []  
        overlayed_images = []
        fill = True

        c = 0 
        p = 0
        inputs = []
        labels = []

        yys = []
        xxs = [] 
        # samples = [samples[56]]
        # embed()
        # for i, s in enumerate(samples):
        #     print(f'{i}, {s}')
        for k, (root, sample, type) in enumerate(samples): 
            path = f'{root}/{sample}' 
            slices_loaded = True
            contours_loaded = True  

            if '126' in path or '072' in path:
                # contours not aligned
                continue 

            if os.path.isdir(path):
                print(path)   
                slices = load_stack(path, scale_factor=scale_factor) 
                 
                if len(slices) == 0: 
                    slices_loaded = False 
                else: 
                    
                    loaded = xml2contours(path, slices)
                    if loaded:
                        stack = Stack(slices, sample, type=type, fill=True) 
                        stack.compute_volumes(filled=True)
                         
 
 
                        dicom_resolution = stack.get_dicom_resolution()
                        gttool_resolution = stack.get_gttool_resolution()
                        if dicom_resolution.shape[0] == 1 and gttool_resolution.shape[0] == 1:
                            stacks += [stack] 
                            p += 1  
                        else:
                            print('incosistant resolution')
                            print(dicom_resolution)
                            print('-')
                            print(gttool_resolution)

                        # image_vol, mask_scar, mask_wall = get_masks(stack, 
 
                    else: 
                        contours_loaded = False  

                
                if not slices_loaded or not contours_loaded:
                    # print(f'{k} {slices_loaded} {contours_loaded} {path}  --------------')
                    c += 1
                # else:
                    
                    # print(f'{k} {slices_loaded} {contours_loaded} {path}')
  




        # all_samples = np.concatenate(overlayed_images, axis=0)
        # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay1_{}.tif'.format(0), all_samples)            surface_points = []

         
        # for i, s in enumerate(stacks):
        #     print(f'{i}, {s.name}')

        
        # from utils.utils_voxel2mesh.file_handle import save_to_obj   
        # outer = stacks[0].all_contours[1]   
        # surface_points =[]
        # resolution = stacks[0].slices[0].dicom_resolution
        # for slice_id, contour in outer.items():    
        #     z = slice_id * torch.ones(contour.shape[0])[:, None]
        #     contour_3d = torch.cat([contour, z], dim=1)
        #     contour_3d = contour_3d * resolution[None]
        #     surface_points += [contour_3d]   
        # surface_points = torch.cat(surface_points, dim=0)
        # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay1.obj', surface_points[None], None, None)


        #  ---
        register = False
        if register:
            dataset = HeartDataset(stacks, cfg, Modes.TRAINING) 
    
            
            configs = load_yaml("./configs/registration_config.yaml") 
            R = Registration(configs=configs,
                            registration_type=configs.CONTOUR_REGISTRATION.type,
                            stacks=stacks,
                            shift_type=configs.CONTOUR_REGISTRATION.shift_type,
                            save_path=configs.PATHS.save_path,
                            save_slices=(configs.PLOT_AND_SAVE.save_unaligned, configs.PLOT_AND_SAVE.save_aligned),
                            plot_contours=configs.PLOT_AND_SAVE.plot_contours)
            R.registration()  
            stacks = dataset.samples   
        #  ---
        # embed()
        # outer = stacks[0].all_contours[1] 
        # centers = stacks[0].all_centers[1] 
        # surface_points =[]
        # resolution = stacks[0].slices[0].dicom_resolution
        # centers_all = []
    

        # for slice_id, contour in outer.items():    
        #     z = slice_id * torch.ones(contour.shape[0])[:, None]  
        #     contour_3d = torch.cat([contour, z], dim=1) 
        #     contour_3d = contour_3d * resolution[None] 

        #     center = centers[slice_id]
        #     z = slice_id * torch.ones(center.shape[0])[:, None] 
        #     center_3d = torch.cat([center, z], dim=1) 
        #     center_3d = center_3d * resolution[None] 
        #     centers_all += [center_3d]  
        #     surface_points += [contour_3d]   
        # surface_points = torch.cat(surface_points, dim=0)
        # centers_all = torch.cat(centers_all, dim=0)
        # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay2.obj', surface_points[None], None, None)
        # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay3.obj', centers_all[None], None, None)

        # embed()

        # img = torch.from_numpy(samples[0].image_vol[..., None]).repeat(1, 1, 1, 3).numpy()
        # img = np.float32(255*(img - img.min())/(img.max()-img.min())) 
        # overly = blend_cpu2(torch.from_numpy(img), torch.from_numpy(samples[0].mask_wall), 3, factor=0.8)
        # mask2 = samples[0].mask_wall[:,70:130,120:190]
        # mask = np.concatenate([mask1, mask2], axis=2)
        # overly = blend_cpu2(torch.from_numpy(img), torch.from_numpy(mask), 3, factor=0.8)
        # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay2.tif', 67 * np.uint8(mask))
         
        # slice.mask_inner_wall = cv2.drawContours(slice.mask_inner_wall, [np.int64(points2d.cpu().numpy())], -1, self.colors[contour.cls], 1)
        print('\nSaving pre-processed data to disk')
        # exit()
        np.random.seed(0)
        perm = np.random.permutation(len(stacks)) 
        
        r = 0.8
        counts = [perm[:np.floor(len(stacks)*r).astype('int')], perm[np.floor(len(stacks)*r).astype('int'):]]
        # counts = [perm, perm]
   
        # v1: original
        # v2: inner/outer class index flipped in y
        # v3: registered stack
        # v4: v2 with 50/50 split
        # v5: with full dataset registered
        # v6: no registration 
        # v7: unsupervised registration, window size: 128
        # v8: center aligned heart
        # v9: v8 with compute center dif and center dist maps
        hists = []
        for i, datamode in enumerate([Modes.TRAINING, Modes.TESTING]):

            samples = [] 

            for j in counts[i]: 
                print('.',end='', flush=True)
                # x = inputs[j]
                # y = labels[j] 
                samples.append(stacks[j])  
            with open(data_root + f'/pre_computed_data_{datamode}_scale_factor_{scale_factor}_r_{r}_v12.pickle', 'wb') as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
 

        print('\nPre-processing complete!')  
 
    def evaluate(self, target, pred, cfg):
        results = {}  
        if target.center is not None: 
            error = 0.0
            y_center = target.center
            yhat_center = pred.center
            count = 0
            for k,v in y_center.items(): 
                error += torch.sqrt(torch.sum((v - yhat_center[k]) ** 2))
                count += 1

            error /= count
            results['mse_error'] = error.cpu().detach()

        if target.voxel is not None:  
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard_lv'] = val_jaccard


        if target.scar_map is not None:  
            if cfg.regress_scar:
                raise('there is a bug')
                yhat_scars = torch.zeros_like(pred.scar_map).long() 
                yhat_scars[pred.scar_map<=2] = 1 # there is a big *****<<<<<<
                yhat_scars[pred.scar_map>2] = 2
                yhat_scars[pred.scar_map>5] = 3

                y_scars = torch.zeros_like(target.scar_map).long()
                y_scars[target.scar_map<=2] = 1
                y_scars[target.scar_map>2] = 2
                y_scars[target.scar_map>5] = 3 
            else:
                yhat_scars = pred.scar_map
                y_scars = target.scar_map
            val_jaccard = jaccard_index(y_scars, yhat_scars, cfg.scar_num_classes)
 
            y_th = []
            yhat_th = []
            for (y_scar_slice, yhat_scar_slice, y_slice, y_hat_slice) in zip(y_scars[0], yhat_scars[0], target.voxel[0], pred.voxel[0]): 
                y_th1_scar = (y_scar_slice==2).sum()
                y_th2_scar = (y_scar_slice==3).sum()
                y_lv = (y_scar_slice>0).sum()
                y_th += [(y_th1_scar, y_th2_scar, y_lv)]

                yhat_th1_scar = (yhat_scar_slice==2).sum()
                yhat_th2_scar = (yhat_scar_slice==3).sum()
                yhat_lv = (yhat_scar_slice>0).sum()
                yhat_th += [(yhat_th1_scar, yhat_th2_scar, yhat_lv)]
                 
            results['y_N'] = y_th
            results['yhat_N'] = yhat_th

            results['jaccard_scar'] = val_jaccard

        # if target.mesh is not None: 
 
        #     point_count = 3000 
 
        #     val_chamfer_weighted_symmetric = np.zeros(cfg.num_classes-1) 
        #     for c in range(cfg.num_classes-1): 
        #         true_mesh = Meshes(verts=target.mesh[c]['vertices'].float().cuda(), faces=target.mesh[c]['faces'].cuda())
        #         pred_mesh = Meshes(verts=pred.mesh[c]['vertices'].float().cuda(), faces=pred.mesh[c]['faces'].cuda())  
        #         true_points = sample_points_from_meshes(true_mesh, point_count)
        #         pred_points = sample_points_from_meshes(pred_mesh, point_count) 
        #         # val_chamfer_weighted_symmetric[c] = torch.sqrt(chamfer_distance(pred_points, true_points)[0]/2)
                
        #         loss, std = chamfer_weighted_symmetric(true_points, pred_points) 
        #         val_chamfer_weighted_symmetric[c] = loss.detach().cpu() 
        #     results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric

        return results

    def update_checkpoint(self, best_so_far, new_value, metric):

        key = metric #'jaccard'
        # key = 'chamfer_weighted_symmetric'
        # key = 'mse_error'
        new_value = new_value[Modes.TESTING][key] 
        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[Modes.TESTING][key]
            return True if np.mean(new_value) > np.mean(best_so_far) else False
            # return True if np.mean(new_value) < np.mean(best_so_far) else False



 
            # x_ = x
            # x_ = np.uint8(255*(x_ - x_.min())/(x_.max()-x_.min()))
            # x_ = y
            # x_ = np.uint8(67*x_)
            # x_ = np.repeat(x_[...,None], 3, axis=3)
            # for slice_Id, contour_2d in contours[1].items():  
            #     x_[slice_Id] = cv2.drawContours(x_[slice_Id], [np.int64(contour_2d.detach().cpu().numpy())], -1, (255, 0, 0), 1)
            # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay2.tif', np.uint8(x_))
            


            # # print(stacks[j].slices[0].image.shape)  
            # y = torch.from_numpy(stacks[j].mask_wall).cuda().long()   
            # x = torch.from_numpy(stacks[j].image_vol).cuda()
            # # x = x - x.min()
            # # x = torch.clamp(x/torch.quantile(x, 0.995), min=0, max=1) 
            # x = x - torch.quantile(x, 0.005)
            # x = torch.clamp(x/torch.quantile(x, 0.995), min=0, max=1) 

            # fg = torch.nonzero(y>0)
            # center = fg.float().mean(dim=0).long()
            # center[0] = x.shape[0]//2
            # center = center.cpu() 

            # patches, contours = crop_images_and_contours([x,y], {}, (32, 128, 128) , tuple(center)) 
            # x, _ = patches 

            # # brightness = brightness_factor*(torch.rand(1).cuda()-0.5)*2
            # # contrast = 0.9 + contrast_factor*torch.rand(1).cuda() 
            # contrast = 1
            # brightness = 0
            # for brightness in np.arange(-0.4, 0.8 , 0.2):
            #     x_ = torch.clamp( contrast*(x - 0.5) + 0.5 + brightness, 0, 1) 
            #     io.imsave(f'/cvlabdata2/cvlab/datasets_udaranga/outputs/tw{brightness}.tif', np.uint8(255*x_.cpu().numpy()))

            # x = x.cpu().numpy()
            # h = np.histogram(x.reshape(-1), 50)
            # hists += [h[0][None]]

            # hists = np.concatenate(hists, axis=0)
            # for i in range(68):
            #     for j in range(50):
            #         print(f'{hists[i,j]} ', end='')
            #     print('')