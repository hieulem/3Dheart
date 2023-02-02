import numpy as np
from skimage import io
from data.data import normalize_vertices, sample_outer_surface_in_voxel, voxel2mesh, clean_border_pixels, normalize_vertices2

import sys
from utils.metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed
from utils.utils_common import crop, Modes, crop_indices, blend
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

class Sample:
    def __init__(self, x, y, resolution):
        self.x = x
        self.y = y
        self.resolution = resolution

class SamplePlus:
    def __init__(self, x, y, y_outer=None, x_super_res=None, y_super_res=None, y_outer_super_res=None, shape=None):
        self.x = x
        self.y = y
        self.y_outer = y_outer
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res  
        self.shape = shape

  
class HeartDataset():

    def __init__(self, data, cfg, mode): 
        self.data = data  

        self.cfg = cfg
        self.mode = mode
 

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx): 
        embed()
        item = self.data[idx]
 
        x = item.x.cuda()
        y = item.y.cuda()    
        res = item.resolution

        y[y==1] = 0
        y[y==2] = 1
        y[y==3] = 2

        mode = self.mode
        config = self.cfg
 
        brightness_factor = 40
        contrast_factor = 0.2
        shift_factor = 20
        x = 255*(x - x.min())/(x.max()-x.min())
        # augmentation done only during training
        if mode == Modes.TRAINING:  # if training do augmentation
            if torch.rand(1)[0] > 0.5:
                x = x.permute([1, 0, 2])
                y = y.permute([1, 0, 2]) 

            if torch.rand(1)[0] > 0.5:
                x = torch.flip(x, dims=[0])
                y = torch.flip(y, dims=[0]) 

            if torch.rand(1)[0] > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1]) 

            if torch.rand(1)[0] > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2]) 

            # Brightness and contrast augmentation
            brightness = -10 + brightness_factor*torch.rand(1).cuda()
            contrast = 0.9 + contrast_factor*torch.rand(1).cuda() 
            x = torch.clamp( contrast*(x - 128) + 128 + brightness, 0, 255)
 
 
        x = (x - x.mean())/x.std()

        x = torch.swapaxes(x[None], 0, 3)[..., 0] 
        y = torch.swapaxes(y[None], 0, 3)[..., 0] 

        y_outer = sample_outer_surface_in_voxel((y>0).long()) 
        surface_points = torch.nonzero(y_outer)
        center = surface_points.float().mean(dim=0).long().cpu()
        if mode == Modes.TRAINING:
            shift = 2*(torch.rand(3) - 0.5)*shift_factor
            shift[2] = 0
            center += shift.long()
        center_x, center_y, center_z = center 
 


        H = W = 128
        D = 64 

        x = crop(x, (D, H, W), (center_x, center_y, center_z))  
        y = crop(y, (D, H, W), (center_x, center_y, center_z))  

        x_high_res = x.clone()[None]
        y_high_res = y.clone()

        scale_factor = (1.0, 1.0, 1.0)
        # scale_factor = (1, 0.5, 0.5)
        x = F.interpolate(x[None, None], scale_factor=scale_factor, mode='trilinear', align_corners=True, recompute_scale_factor=False)[0]
        y = F.interpolate(y[None, None], scale_factor=scale_factor, mode='nearest', recompute_scale_factor=False)[0, 0].long() 

        surface_points_normalized_all = []
        vertices_mc_all = []
        faces_mc_all = [] 

        y_temp = y.clone()
        y = y_high_res
        for i in range(1, config.num_classes):   
            shape = torch.tensor(y.shape)[None].float()
            if mode != Modes.TRAINING:
                gap = 1
                y_ = clean_border_pixels((y==i).long(), gap=gap)
                vertices_mc, faces_mc = voxel2mesh(y_, gap, shape)
                vertices_mc_all += [vertices_mc]
                faces_mc_all += [faces_mc]
        
            if i == 1:
                y_outer = sample_outer_surface_in_voxel((y>0).long()) 
            else:
                y_outer = sample_outer_surface_in_voxel((y==i).long()) 
            # y_outer = sample_outer_surface_in_voxel((y==i).long()) 
             
            surface_points = torch.nonzero(y_outer)
            surface_points = normalize_vertices2(surface_points, shape.to(surface_points.device)) 
            surface_points_normalized = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z 
            # surface_points_normalized = y_outer 
        
        
            perm = torch.randperm(len(surface_points_normalized))
            point_count = 3000
            surface_points_normalized_all += [surface_points_normalized[perm[:np.min([len(perm), point_count])]].cuda()]  # randomly pick 3000 points
        
        y = y_temp
        if mode == Modes.TRAINING:
            return {   'x': x,  
                    'x_high_res': x_high_res,
                    'y_voxels': y, 
                    'surface_points': surface_points_normalized_all, 
                    'unpool':[0, 1, 0, 1, 0]
                    }
        else:
            return {   'x': x, 
                    'y_voxels': y, 
                    'x_high_res': x_high_res,
                    'y_voxels_high_res': y_high_res,
                    'vertices_mc': vertices_mc_all,
                    'faces_mc': faces_mc_all,
                    'surface_points': surface_points_normalized_all, 
                    'resolution': res,
                    'unpool':[0, 1, 0, 1, 1]}
  

class Heart():




    def pick_surface_points(self, y_outer, point_count):
        idxs = torch.nonzero(y_outer) 
        perm = torch.randperm(len(idxs))

        y_outer = y_outer * 0  
        idxs = idxs[perm[:point_count]]
        y_outer[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
        return y_outer

    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.patch_shape

        data_root = cfg.dataset_path
        data = {}
        for i, datamode in enumerate([Modes.TRAINING, Modes.TESTING]):
            with open(f'{data_root}/pre_computed_data_{datamode}.pickle', 'rb') as handle:
                samples = pickle.load(handle) 
                data[datamode] = HeartDataset(samples, cfg, datamode) 

        return data

    def pre_process_dataset(self, cfg):
        '''
         :
        ''' 

        data_root = cfg.dataset_path
        inputs = []
        labels = []
        dims = []
        resolutions = []
        for i in range(1, 101):

            j = 4 if i == 90 else 1
            image_path = f'{data_root}/patient{i:03d}/patient{i:03d}_frame0{j}.nii.gz' 
            label_path = f'{data_root}/patient{i:03d}/patient{i:03d}_frame0{j}_gt.nii.gz' 
            nimg = nib.load(image_path)
            nlabel = nib.load(label_path)
 
 
            x = nimg.get_data()
            x = np.float32(x) 
            # mean_x = np.mean(x)
            # std_x = np.std(x) 
            # x = (x - mean_x)/std_x
            x = torch.from_numpy(x) 
            inputs += [x]

            y = nlabel.get_data()
            y = torch.from_numpy(y) 
            labels += [y]
            dims += [np.array(nimg.get_data().shape)[None]]

            resolutions += [torch.from_numpy(nimg.header['pixdim'][1:4])]
            # nimg.get_data(), nimg.affine, nimg.header

        dims = np.concatenate(dims, axis=0) 
 

        print('\nSaving pre-processed data to disk')
        np.random.seed(0)
        perm = np.random.permutation(len(inputs)) 
        counts = [perm[:len(inputs)//2], perm[len(inputs)//2:]]
   
        for i, datamode in enumerate([Modes.TRAINING, Modes.TESTING]):

            samples = [] 

            for j in counts[i]: 
                print('.',end='', flush=True)
                x = inputs[j]
                y = labels[j]
                res = resolutions[j]
 
                samples.append(Sample(x, y, res)) 

            with open(data_root + f'/pre_computed_data_{datamode}.pickle', 'wb') as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
        
        print('Pre-processing complete')  
 
    def evaluate(self, target, pred, cfg):
        results = {}


        if target.voxel is not None: 
             
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard'] = val_jaccard

        if target.mesh is not None:
            target_points = target.points
            pred_points = pred.mesh
            val_chamfer_weighted_symmetric = np.zeros(len(target_points))

            for i in range(len(target_points)):
                val_chamfer_weighted_symmetric[i] = chamfer_weighted_symmetric(target_points[i].cpu(), pred_points[i]['vertices'])

            results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric

        return results

    def update_checkpoint(self, best_so_far, new_value):

        key = 'jaccard'
        new_value = new_value[Modes.TESTING][key]

        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[Modes.TESTING][key]
            return True if np.mean(new_value) > np.mean(best_so_far) else False



 