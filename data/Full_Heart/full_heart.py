from tkinter import W
from data.CHUV.CHUV3 import HeartDataset
from data.CHUV.register import Registration
from data.data import clean_border_pixels, normalize_vertices, sample_outer_surface_in_voxel, voxel2mesh 
import numpy as np

from utils.metrics import jaccard_index, chamfer_weighted_symmetric 
from utils.utils_common import Modes, crop, crop_images_and_contours, load_yaml, mkdir, permute 
from torchmcubes import marching_cubes
from utils import stns
# from utils.utils_mesh import sample_outer_surface, get_extremity_landmarks, voxel2mesh, clean_border_pixels, sample_outer_surface_in_voxel, normalize_vertices 

 
# from utils import stns
from torch.utils.data import Dataset 
import pickle 
import torch 
import os 
import nibabel as nib
from IPython import embed 
from data.CHUV.load_stack import load_snapshots, load_stack, Stack, xml2contours
import cv2 as cv
from skimage import io 
from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
import torch.nn.functional as F

from utils.utils_common import DataModes
from utils.utils_voxel2mesh.file_handle import save_to_obj

class Sample:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name 

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

        contrast_factor = 0.2
        brightness_factor = 0.2

        x = self.samples[idx].x.cuda().float() 
        y = self.samples[idx].y.cuda().long()
        name = self.samples[idx].name

        # Make it portrait 
        # (x, y), contours = permute([x, y], {})        
        x = x - x.min()
        x = torch.clamp(x/torch.quantile(x, 0.995), min=0, max=1)

        fg = torch.nonzero(y>0)
        center = fg.float().mean(dim=0).long()
        # center[0] = x.shape[0]//2
        center = center.cpu() 

        if self.mode == Modes.TRAINING:
            # random crop during training
            shift = 2*(torch.rand(3) - 0.5)*self.cfg.shift_factor
            shift[0] = 0
            center += shift.long() 
 
        x = crop(x, self.cfg.patch_shape, center) 
        y = crop(y, self.cfg.patch_shape, center)   

        x = x[None]
        # augmentation done only during training
        if self.mode == Modes.TRAINING:  # if training do augmentation
            if torch.rand(1)[0] > 0.5:
                x = x.permute([0, 1, 3, 2])
                y = y.permute([0, 2, 1]) 

            if torch.rand(1)[0] > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[0]) 

            if torch.rand(1)[0] > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[1]) 

            if torch.rand(1)[0] > 0.5:
                x = torch.flip(x, dims=[3])
                y = torch.flip(y, dims=[2]) 

            # Brightness and contrast augmentation
            brightness = brightness_factor*(torch.rand(1).cuda()-0.5)*2
            contrast = 0.9 + contrast_factor*torch.rand(1).cuda() 
            x = torch.clamp( contrast*(x - 0.5) + 0.5 + brightness, 0, 1)

            # orientation = torch.tensor([0, -1, 0]).float()
            # new_orientation = (torch.rand(3) - 0.5) * 2 * np.pi 
            # new_orientation = F.normalize(new_orientation, dim=0)
            # q = orientation + new_orientation
            # q = F.normalize(q, dim=0)
            # theta_rotate = stns.stn_quaternion_rotations(q)

            # shift = torch.tensor([d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * self.config.augmentation_shift_range, y.shape)])
            # theta_shift = stns.shift(shift)
            
            # f = 0.1
            # scale = 1.0 - 2 * f *(torch.rand(1) - 0.5) 
            # theta_scale = stns.scale(scale) 

            # theta = theta_rotate @ theta_shift @ theta_scale 
            # x, y, y_outer = stns.transform(theta, x, y, y_outer) 
    
        x = torch.cat([x, self.base_grid], dim=0) 

        surface_points_normalized_all = []
        vertices_mc_all = []
        faces_mc_all = [] 
        for i in range(1, self.cfg.num_classes):   
            shape = torch.tensor(y.shape)[None].float()
            if self.mode != Modes.TRAINING:
                gap = 1
                y_ = clean_border_pixels((y==i).long(), gap=gap)
                vertices_mc, faces_mc = voxel2mesh(y_, gap, shape)
                vertices_mc_all += [vertices_mc]
                faces_mc_all += [faces_mc]
         
            y_outer = sample_outer_surface_in_voxel((y==i).long()) 
            surface_points = torch.nonzero(y_outer)
            surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
            surface_points_normalized = normalize_vertices(surface_points, shape) 
            # surface_points_normalized = y_outer 
         
            perm = torch.randperm(len(surface_points_normalized))
            point_count = 3000
            surface_points_normalized_all += [surface_points_normalized[perm[:np.min([len(perm), point_count])]].cuda()]  # randomly pick 3000 points
         
        if self.mode == Modes.TRAINING:
            return {   'x': x,  
                    'y_voxels': y, 
                    'surface_points': surface_points_normalized_all, 
                    'unpool':[0, 1, 0, 1, 0]
                    }
        else:
            return {   'x': x, 
                    'y_voxels': y, 
                    'vertices_mc': vertices_mc_all,
                    'faces_mc': faces_mc_all,
                    'name': name,
                    'surface_points': surface_points_normalized_all, 
                    'unpool':[0, 1, 0, 1, 1]}
   

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
        r = 0.75
        scale_factor=2
        for i, datamode in enumerate([Modes.TRAINING, Modes.TESTING]):
            path = f'{data_root}/pre_computed_data_{datamode}_scale_factor_{scale_factor}_r_{r}_v2.pickle'
            print(path)
            with open(path, 'rb') as handle:
                samples = pickle.load(handle) 
                data[datamode] = HeartDataset(samples, cfg, datamode) 

        return data
 

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
        

        scale_factor = 2
        data_root = cfg.dataset_path  

        

        samples = []
        # for i in range(1, 21):
        #     image = f'{data_root}/mr_train/mr_train_10{i:02d}_image.nii.gz'
        #     label = f'{data_root}/mr_train/mr_train_10{i:02d}_label.nii.gz'
        #     samples += [(image, label)]

        for i in range(1, 41):
            image = f'{data_root}/mr_test/mr_test_20{i:02d}_image.nii.gz'
            label = f'{data_root}/mr_test/mr_test_20{i:02d}_label_encrypt_1mm.nii.gz'
            samples += [(image, label)]

        images = []
        labels = []
        names = []

        colors= [(0, 0, 255), (0, 255, 0), (255, 0, 0), (127, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255) ]
        xcs = []
        resolutions = []
        # for i in range(1, 21):
        for sample in samples:
            # print(f'{data_root}/mr_train_10{i:02d}_image.nii.gz')
            # nimg = nib.load(f'{data_root}/mr_train_10{i:02d}_image.nii.gz')
            # nlabel = nib.load(f'{data_root}/mr_train_10{i:02d}_label.nii.gz')
            nimg_name, nlabel_name = sample
            nimg = nib.load(nimg_name)
            nlabel = nib.load(nlabel_name)

            embed()

            x = nimg.get_data()
            x = np.float32(x) 
            # mean_x = np.mean(x)
            # std_x = np.std(x) 
            # x = (x - mean_x)/std_x
            x = torch.from_numpy(x)
            # x = (x - x.min())/(x.max()-x.min())  
            x = torch.swapaxes(x, 0, 2) 

            y_ = nlabel.get_data()
            y_ = torch.from_numpy(np.int64(y_)).float()

            for kk in range(146):
                print(f'{y_[:,:,kk].mean().item()},',end='')
            y_[y_>1000] = 0

            io.imsave(f'/cvlabdata2/cvlab/datasets_udaranga/outputs/imgs.tif', xcs_) 

            y_ = torch.swapaxes(y_, 0, 2)  
            y = torch.zeros_like(y_).long()
            y[y_==205] = 1
            y[y_==420] = 2
            y[y_==421] = 2
            y[y_==500] = 1
            y[y_==550] = 3
            y[y_==600] = 4
            y[y_==820] = 5
            y[y_==850] = 6
 

            w_resolution, h_resolution, d_resolution = torch.from_numpy(nimg.header['pixdim'][1:4])
             

            D, H, W = x.shape
            D = int(D * d_resolution)//scale_factor #  
            H = int(H * h_resolution)//scale_factor # 
            W = int(W * w_resolution)//scale_factor  #  
            # we resample such that 1 pixel is 1 mm in x,y and z directiions
            base_grid = torch.zeros((1, D, H, W, 3))
            w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
            d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)
            base_grid[:, :, :, :, 0] = w_points
            base_grid[:, :, :, :, 1] = h_points
            base_grid[:, :, :, :, 2] = d_points
            
            grid = base_grid.cuda()
                 
            x = F.grid_sample(x[None, None].cuda(), grid, mode='bilinear', padding_mode='border', align_corners=True)[0, 0].cpu() 
            y = F.grid_sample(y[None, None].cuda().float(), grid, mode='nearest', padding_mode='border', align_corners=True)[0, 0].long().cpu() 
            
            resolutions += [torch.from_numpy(nimg.header['pixdim'][1:4])]
             
            images += [x]
            labels += [y]
            names += [f'mr_train_10{i:02d}_image.nii.gz']

            # xc = np.uint8(255 * x.clone()[..., None].repeat(1,1,1,3))
            # mkdir(f'/cvlabdata2/cvlab/datasets_udaranga/outputs/meshes') 
            # mkdir(f'/cvlabdata2/cvlab/datasets_udaranga/outputs/meshes/{i}') 
            # for k in range(1, 7):
            #     y_k = (y==k).cuda().contiguous()
            #     verts, faces = marching_cubes(y_k.float(), 0.5)  
            #     save_to_obj(f'/cvlabdata2/cvlab/datasets_udaranga/outputs/meshes/{i}/mesh_{i}_cls_{k}.obj', verts[None], faces[None], None)

            #     for z, slice in enumerate(y_k.cpu().numpy()):
            #         _, contours, _ = cv.findContours(np.uint8(slice>0), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                     
            #         for _, ctrs in enumerate(contours):   
            #             ctrs = np.squeeze(ctrs)  
            #             if len(ctrs.shape) == 2:
            #                 xc[z] = cv.drawContours(xc[z], [ctrs], -1, colors[k], 1)  
            
            # D, H, W, _ = xc.shape
            # xc = crop(xc, (D, 400, 400, 3), (D//2, H//2, W//2, 1), mode='constant')
            # xcs += [xc] 
  
        # xcs_ = np.concatenate(xcs, axis=0)
        # io.imsave(f'/cvlabdata2/cvlab/datasets_udaranga/outputs/imgs.tif', xcs_) 
          
 
        # slice.mask_inner_wall = cv2.drawContours(slice.mask_inner_wall, [np.int64(points2d.cpu().numpy())], -1, self.colors[contour.cls], 1)
        print('\nSaving pre-processed data to disk')
        # exit() 
        np.random.seed(0)
        perm = np.random.permutation(len(images)) 
        
        r = 0.75
        counts = [perm[:np.floor(len(images)*r).astype('int')], perm[np.floor(len(images)*r).astype('int'):]]
        # counts = [perm, perm]
   
        # v1: original  
        for i, datamode in enumerate([Modes.TRAINING, Modes.TESTING]):

            samples = []  
            print('---')
            for j in counts[i]:  
                print(images[j].shape)
                samples.append(Sample(images[j], labels[j], names[j]))  
            with open(data_root + f'/pre_computed_data_{datamode}_scale_factor_{scale_factor}_r_{r}_v2.pickle', 'wb') as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
 

        print('\nPre-processing complete!')  
 
    def evaluate(self, target, pred, cfg):
        results = {}   

        if target.voxel is not None:  
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard'] = val_jaccard

 
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