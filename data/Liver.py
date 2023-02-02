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
    def __init__(self, x, y, resolution, name=''):
        self.x = x
        self.y = y
        self.resolution = resolution
        self.name = name

class SamplePlus:
    def __init__(self, x, y, y_outer=None, x_super_res=None, y_super_res=None, y_outer_super_res=None, shape=None):
        self.x = x
        self.y = y
        self.y_outer = y_outer
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res  
        self.shape = shape

  
class LiverDataset():

    def __init__(self, data, cfg, mode): 
        self.data = data  

        self.cfg = cfg
        self.mode = mode
 

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx): 
        item = self.data[idx]
 
        x = item.x.cuda()
        y = item.y.cuda()    
        res = item.resolution
        name = item.name

        # y[y>0] = 1
        y[y==1] = 0
        y[y==2] = 1
        mode = self.mode
        config = self.cfg
 
        brightness_factor = 40
        contrast_factor = 0.2
        shift_factor = 5
        x = 255*(x - x.min())/(x.max()-x.min())
        # augmentation done only during training
        if mode == Modes.TRAINING:  # if training do augmentation
            if torch.rand(1)[0] > 0.5:
                x = x.permute([0, 2, 1])
                y = y.permute([0, 2, 1]) 

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

        y_outer = sample_outer_surface_in_voxel((y>0).long()) 
        surface_points = torch.nonzero(y_outer)
        center = surface_points.float().mean(dim=0).long().cpu()
        if mode == Modes.TRAINING:
            shift = 2*(torch.rand(3) - 0.5)*shift_factor
            shift[2] = 0
            center += shift.long()
        center_x, center_y, center_z = center 


        H = W = 64
        D = 64 

        x = crop(x, (D, H, W), (center_x, center_y, center_z))  
        y = crop(y, (D, H, W), (center_x, center_y, center_z))  

        x_high_res = x.clone()[None]
        y_high_res = y.clone()

        scale_factor = (1.0, 1.0, 1.0)
        
        # scale_factor = (1, 0.5, 0.5)
        x = F.interpolate(x[None, None], scale_factor=scale_factor, mode='trilinear', align_corners=True, recompute_scale_factor=False)[0]
        y = F.interpolate(y[None, None].float(), scale_factor=scale_factor, mode='nearest', recompute_scale_factor=False)[0, 0].long() 

        surface_points_normalized_all = []
        vertices_mc_all = []
        faces_mc_all = [] 

        y_temp = y.clone()
        y = y_high_res
        for i in range(1, config.num_classes):   
            shape = torch.tensor(y.shape)[None].float()
            if mode != Modes.TRAINING:
                gap = 1
                y_ = clean_border_pixels((y>0).long(), gap=gap)
                vertices_mc, faces_mc = voxel2mesh(y_, gap, shape)
                vertices_mc_all += [vertices_mc]
                faces_mc_all += [faces_mc]
        
            # if i == 1:
            #     y_outer = sample_outer_surface_in_voxel((y>0).long()) 
            # else:
            #     y_outer = sample_outer_surface_in_voxel((y==i).long())  
            y_outer = sample_outer_surface_in_voxel((y==i).long()) 
             
            surface_points = torch.nonzero(y_outer)
            surface_points = normalize_vertices2(surface_points, shape.to(surface_points.device)) 
            surface_points_normalized = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z 
            # surface_points_normalized = y_outer 
         
            perm = torch.randperm(len(surface_points_normalized))
            point_count = 3000
            sampled_points = surface_points_normalized[perm[:np.min([len(perm), point_count])]].cuda()
            surface_points_normalized_all += [sampled_points]  # randomly pick 3000 points
        
        y = y_temp
        if mode == Modes.TRAINING:
            return {   'x': x,  
                    'x_high_res': x_high_res,
                    'y_voxels': y, 
                    'y_voxels_high_res': y_high_res, 
                    'surface_points': surface_points_normalized_all, 
                    'resolution': res,
                    'name': name,
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
                    'name': name,
                    'unpool':[0, 1, 0, 1, 1]}
  

class Liver():




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
                data[datamode] = LiverDataset(samples, cfg, datamode) 

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
        file_name = []
        
        sizes = []
        for i in range(131): 
            image_path = f'{data_root}/imagesTr/liver_{i:d}.nii.gz'  
            label_path = f'{data_root}/labelsTr/liver_{i:d}.nii.gz'   
            nimg = nib.load(image_path)
            nlabel = nib.load(label_path)
            # embed()
 
            x = nimg.get_data()
            x = np.float32(x) 
            # mean_x = np.mean(x)
            # std_x = np.std(x) 
            # x = (x - mean_x)/std_x
            x = torch.from_numpy(x)

            y = nlabel.get_data()
            y = torch.from_numpy(y)  

            x = x.cuda()
            y = y.cuda()

            a = (y==2).sum().item()
            sizes += [a]
            print(a)
            if a>251000: 
                x = 255*(x - x.min())/(x.max()-x.min())
 
                x = torch.swapaxes(x[None], 0, 3)[..., 0] 
                x = torch.swapaxes(x, 1, 2) 
                x = torch.flip(x, dims=[1])
                
                y = torch.swapaxes(y[None], 0, 3)[..., 0] 
                y = torch.swapaxes(y, 1, 2) 
                y = torch.flip(y, dims=[1])
                

                y_outer = sample_outer_surface_in_voxel((y>0).long()) 
                surface_points = torch.nonzero(y_outer)
                center = surface_points.float().mean(dim=0).long().cpu()

                center_x, center_y, center_z = center 
        


                H = W = 512
                D = 384 

                x = crop(x, (D, H, W), (center_x, center_y, center_z))  
                y = crop(y, (D, H, W), (center_x, center_y, center_z))              


                scale_factor = (0.125, 0.125, 0.125)

                x = F.interpolate(x[None, None], scale_factor=scale_factor, mode='trilinear', align_corners=True, recompute_scale_factor=False)[0, 0].cpu()
                y = F.interpolate(y[None, None], scale_factor=scale_factor, mode='nearest', recompute_scale_factor=False)[0, 0].long().cpu()
    
                # embed()
                # io.imsave('/cvlabdata2/home/wickrama/projects/annotator-mitk/interactive-annotation-mitk/server/outputs/overlay_y_hat.tif', np.uint8(x.cpu().numpy()))
                # io.imsave('/cvlabdata2/home/wickrama/projects/annotator-mitk/interactive-annotation-mitk/server/outputs/overlay_y.tif', np.uint8(67*y.float().cpu().numpy()))
 
            
                inputs += [x]
                labels += [y]
                dims += [np.array(nimg.get_data().shape)[None]]
                file_name += [f'liver_{i:d}.nii.gz']
            # print(y.shape)

                resolutions += [torch.from_numpy(nimg.header['pixdim'][1:4])]
            # nimg.get_data(), nimg.affine, nimg.header
         

        dims = np.concatenate(dims, axis=0) 
        print(f'Length: {len(labels)}')
        # embed()
        # vals = []
        # for y in labels:
        #     nonzero = y.sum(dim=[0,1]).nonzero()[:, 0]
        #     clip_min = nonzero.min()
        #     clip_max = nonzero.max() + 1
        #     length = (clip_max - clip_min)
        #     vals += [length.item()] 
        # vals = torch.tensor(vals)
        
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
                name = file_name[j]
 
                samples.append(Sample(x, y, res, name)) 

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
                
                if target_points[i].shape[1] > 0:
                    # print(target_points[i].shape)
                    
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



 