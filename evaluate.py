from data.Full_Heart.predict import predict, save_results_full_hearts
from data.data import sample_outer_surface_in_voxel
from utils.registration_utils import shift_2d_replace
from utils.utils_common import Modes, blend_cpu3, gaussian_filter, hist_normalize, mkdir, blend, crop_indices, blend_cpu, append_line, val2str, write_lines, blend_cpu2
from utils.utils_voxel2mesh.file_handle import get_slice_mesh, save_to_obj, save_to_texture_obj  
from torch.utils.data import DataLoader
import numpy as np
import torch
from skimage import io 
import itertools
import torch.nn.functional as F
import os
from scipy import ndimage
from IPython import embed
import wandb
# from utils.rasterize.rasterize import Rasterize
# from utils import stns
from pytorch3d.structures import Meshes
from utils.rasterize.rasterize2 import rasterize_vol
import cv2 as cv
from data.CHUV.load_stack import resacle, resacle_z
from torchmcubes import marching_cubes
from collect_env import run
import trimesh
import csv
import igl
import pyvista as pv
from pytorch3d.ops import sample_points_from_meshes
from pyremesh import remesh_botsch
from skimage import measure
import torch.nn as nn

class Structure(object):

    def __init__(self, 
                voxel=None, 
                mesh=None, 
                contours2d=None, 
                contours3d=None, 
                slices=None, 
                name=None, 
                surface_points=None, 
                center=None, 
                center_map=None, 
                distance_map=None, 
                scar_mesh=None, 
                scar_map=None, 
                myocardium=None,
                mu_scar=None,
                std_scar=None):
        self.voxel = voxel 
        self.mesh = mesh   
        self.contours2d = contours2d
        self.contours3d = contours3d
        self.surface_points = surface_points
        self.name = name
        self.slices = slices
        self.center = center
        self.center_map = center_map
        self.distance_map = distance_map
        self.scar_map = scar_map
        self.scar_mesh = scar_mesh
        self.myocardium = myocardium
        self.mu_scar = mu_scar
        self.std_scar = std_scar
 
def write_to_wandb(split, performences): 
    log_vals = {}
    for key, value in performences[split].items(): 
        log_vals[split + '_' + key + '/mean'] = np.mean(performences[split][key])  
        num_classes = performences[split][key].shape[-1]
        for i in range(num_classes):
            log_vals[split + '_' + key + '/class_' + str(i+1)] = np.mean(performences[split][key][:, i]) 
    try:
        wandb.log(log_vals)
    except:
        print('')


class Evaluator(object):
    def __init__(self, net, optimizer, data, save_path, config, support):
        self.data = data
        self.net = net
        self.current_best = None
        self.save_path = save_path + '/best_performance3' 
        self.latest = save_path + '/latest' 
        self.optimizer = optimizer
        self.config = config
        self.support = support
        self.count = 0 


    def save_model(self, epoch):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.save_path + '/model.pth')


    def evaluate(self, epoch, writer=None, backup_writer=None):
        # self.net.eval()
        
        performences = {}
        predictions = {}

        modes_to_eval = [Modes.TESTING]
        for split in modes_to_eval: 
            dataloader = DataLoader(self.data[split], batch_size=1, shuffle=False) 
            performences[split], predictions[split] = self.evaluate_set(dataloader)

            write_to_wandb(split, performences)
         
        if self.support.update_checkpoint(best_so_far=self.current_best, new_value=performences, metric=self.config.eval_metric):
        # if True:  
            print('updating checkpoint...') 
            if self.config.trial_id is None: 
                mkdir(self.save_path) 
                mkdir(self.save_path + '/mesh')
                mkdir(self.save_path + '/points')
                mkdir(self.save_path + '/voxels') 
                self.save_model(epoch)
                for mode in modes_to_eval:
                    self.save_results(predictions[mode], epoch, performences[mode], self.save_path, f'/{mode}_', self.config)
                self.current_best = performences
            else: 
                mkdir(self.save_path + '_') 
                mkdir(self.save_path + '_/mesh')
                mkdir(self.save_path + '_/points')
                mkdir(self.save_path + '_/voxels')  
                for mode in modes_to_eval:
                    self.save_results(predictions[mode], epoch, performences[mode], self.save_path + '_', f'/{mode}_', self.config) 
            print('Saving complete!')
        print('Eval done!')
        self.net.eval()
        self.net.train()

    def predict(self, data, config):
        name = config.name
        if name == 'unet':
            self.save_results = self.save_results_unet
            name = data['name']
            y_hat, _ = self.net(data) 
            y_hat = torch.argmax(y_hat, dim=1).cpu()

            x = data['x'].cpu()
            y = Structure(voxel=data['y_scar'].cpu(), name=name)
            y_hat = Structure(voxel=y_hat)

        elif name == 'full_heart':
            self.save_results = save_results_full_hearts
            x, y, y_hat = predict(self.net, data, config)

        elif name == 'voxel2mesh':  
            self.save_results = self.save_results_v2m
            scale_factor = config.scale_factor
            x = data['x']
            # x = data['x_high_res'] 
            # y_ = data['y_voxels']
            name = data['name']
            resolution = data['resolution'].cpu() 

            pred, data = self.net(data, mode=Modes.TESTING) 
            pred_localizer = pred['register']
            pred_voxel2mesh = pred['voxel2mesh']
            pred_scarnet = pred['scarnet']

            x = data['x'] 
            y = data['y_voxels'] 

            pred_meshes = []
            true_meshes = []
            true_contours2d = []
            pred_contours2d = []
            true_contours3d = []
            pred_contours3d = [] 
            pred_scar_meshes = []
            true_scar_meshes = []

            nonzero = y.cpu().sum(dim=[2,3]).nonzero()[:,1] 
            clip_min = nonzero.min() if y.sum() > 0 else 0
            clip_max = (nonzero.max() + 1) if y.sum() > 0 else y.shape[1]
  
            shape = torch.tensor(x.shape)  
            shape = shape[2:].flip([0]) # flip() because we flip the ijk values in data laoder 
            # shape = torch.tensor(x.shape[2:]).flip(0)[None, None].float().cuda()
            
            x_original = x.clone()  
            x = torch.from_numpy(hist_normalize(x[0, 0].cpu().numpy())).cuda().float()
            # x = torch.from_numpy(x[0, 0].cpu().numpy()).cuda().float()
            x = (x-x.min())/(x.max()-x.min())
            x = resacle(x, scale_factor=scale_factor)[None, None].cpu()

            slice_grid, slice_grid_uv, grid_faces = get_slice_mesh(x, scale_factor) 
 
            slices = {}
            for slice_id in range(clip_min, clip_max):
                z = slice_id * torch.ones(slice_grid.shape[0])[:, None]
                grid = torch.cat([slice_grid, z], dim=1)[None] 
                grid = grid * resolution
                slices[slice_id] = {'slice_vertices':grid, 'slice_uv':slice_grid_uv, 'slice_faces': grid_faces, 'image':x[0, 0, slice_id]}

            

            
            pred_voxels_rasterized_all = []
            true_voxels_all = []

            surface_points_all = []

            for c in range(self.config.lv_num_classes-1):  
                # embed()

                pred_vertices = pred_voxel2mesh[c][-1][0].detach().data.cpu()
                pred_faces = pred_voxel2mesh[c][-1][1].detach().data.cpu()
                pred_probs = pred_voxel2mesh[c][-1][3].detach().data.cpu()   
                pred_meshes += [{'vertices': (pred_vertices/2 + 0.5) * (shape-1) * resolution, 'faces':pred_faces, 'normals':None}] 
                
                
                if len(data['true_verts']) > 0:
                    true_vertices = data['true_verts'][c].data.cpu()  
                    true_faces = data['true_faces'][c].data.cpu()  
                    true_contour2d = data['true_contours'][c]  # True contour is in non-scaled x coordinates
                    surface_points = data['surface_points'][c] 
                    surface_points_all += [(surface_points.cpu()/2 + 0.5) * (shape-1) * resolution]
                    true_meshes += [{'vertices': (true_vertices/2 + 0.5) * (shape-1) * resolution, 'faces':true_faces, 'normals':None}] 

                    true_contours2d += [true_contour2d] 

                    true_voxels_ = np.uint8(torch.zeros_like(x[0, 0]).numpy())  
                    true_contour3d = [] 
                    for slice_id, contour in true_contour2d.items():   
                        contour = contour.numpy()[0]
                        true_voxels_[slice_id] = cv.fillPoly(true_voxels_[slice_id], [np.int64(scale_factor * contour)], (1, 0, 0))
                        # true_voxels_[z] = cv.drawContours(true_voxels_[z], [scale_factor * contour], -1, (1, 0, 0), 1) 

                        z = slice_id * np.ones(contour.shape[0])[:, None]
                        contour_3d = np.concatenate([contour, z], axis=1)
                        contour_3d = contour_3d[None] * resolution.numpy()
                        true_contour3d += [contour_3d]   
                    true_voxels_all += [torch.from_numpy(true_voxels_)[None].long()] 

                    true_contour3d = torch.from_numpy(np.concatenate(true_contour3d, axis=1))
                    N = true_contour3d.shape[1] 
                    indices = torch.arange(N)[:, None]
                    true_contour3d_faces = torch.cat([indices, indices, torch.roll(indices, dims=0, shifts=[1])], dim=1)[None]
                    true_contours3d += [{'vertices': true_contour3d, 'faces':true_contour3d_faces, 'normals':None}] 

                else:
                    true_vertices = true_faces = true_contour2d = surface_points = true_contour3d = true_voxels_all = true_contours2d = true_meshes = true_contours3d = None
                    # true_meshes += [{'vertices': None, 'faces':None, 'normals':None}]  
                    # true_contours3d += [{'vertices': None, 'faces':None, 'normals':None}]  
                


                verts = torch.flip(pred_vertices, [2]) 
                faces = pred_faces   
                mesh = Meshes(verts=verts.detach(), faces=faces.detach().long()) 
                pred_voxels_rasterized = rasterize_vol(mesh, x.shape[2:])   
                pred_voxels_rasterized_all += [pred_voxels_rasterized[None].cpu()] 
                  
                # Pred contour is in non-scaled x coordinates
                pred_contour2d = {}
                pred_contour3d = []
                for slice_id, slice in enumerate(pred_voxels_rasterized.numpy()):
                    _, contours, _ = cv.findContours(np.uint8(slice>0), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    for k, ctrs in enumerate(contours):  
                        if k == 0:
                            pred_contour2d[slice_id] = []
                        ctrs = np.squeeze(ctrs)/scale_factor
                        pred_contour2d[slice_id] += [ctrs] 

                        if len(ctrs.shape) == 1: 
                            z = slice_id * np.ones(1)[:, None] 
                            ctrs = ctrs[None] 
                        else:
                            z = slice_id * np.ones(ctrs.shape[0])[:, None] 

                        contour_3d = np.concatenate([ctrs, z], axis=1)
                        contour_3d = contour_3d[None] * resolution.numpy()
                        pred_contour3d += [contour_3d]
                        # canvas[z] = cv.drawContours(slice, [ctrs], -1, (128, 128, 0), 1) 
 
                pred_contours2d += [pred_contour2d]   

                pred_contour3d = torch.from_numpy(np.concatenate(pred_contour3d, axis=1))
                N = pred_contour3d.shape[1] 
                indices = torch.arange(N)[:, None]
                pred_contour3d_faces = torch.cat([indices, indices, torch.roll(indices, dims=0, shifts=[1])], dim=1)[None] 
                pred_contours3d += [{'vertices': pred_contour3d, 'faces':pred_contour3d_faces, 'normals':None}]    

                # scar extraction
 
            pred_voxels = torch.zeros_like(x)[:,0].long()
            pred_voxels[pred_voxels_rasterized_all[1]==1] = 2 # outer
            pred_voxels[pred_voxels_rasterized_all[0]==1] = 1 # inner   
            pred_voxels = pred_voxels.cpu()  

            true_voxels = torch.zeros_like(x)[:,0].long()
            if y.sum() > 0:
                true_voxels[true_voxels_all[1]==1] = 2 # outer
                true_voxels[true_voxels_all[0]==1] = 1 # inner          

            # -----------------------
            # Scar extraction  
            # -----------------------   
             
            yhat_scars = torch.argmax(pred_scarnet, dim=1) 
            # yhat_scars = torch.zeros_like(data['y_scar'])
            # for i in range(1, self.config.nu m_classes):
            #     yhat_scars[pred_scarnet[:,i]>0] = i+1
            y_scars = data['y_scar']
            
            x_scar = x_original.clone()[0,0]
            x_scar[yhat_scars[0]!=1] = 0
            
            mu_scar = torch.sum(x_scar, dim=[1,2])/torch.sum(yhat_scars[0]==1, dim=[1,2])

            std_scar = (x_original.clone()[0,0]-mu_scar[:, None, None]) ** 2
            std_scar[yhat_scars[0]!=1] = 0
            std_scar = torch.sqrt(torch.sum(std_scar, dim=[1,2])/(torch.sum(yhat_scars[0]==1, dim=[1,2])-1))
               
                

            y_myocardium = data['y_myocardium']  

            val = 1
            valz = 9
            # val = 3
            # valz = 3
            filter_dim = (valz,val,val)
            filter_pad = (valz//2, val//2, val//2)
            # filter = gaussian_filter(filter_dim, sigma=1)
            # filter_op = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=filter_dim, padding=filter_pad)
            # filter_op.weight.data = filter[None, None] 
            # filter_op.bias.data.fill_(0.0)
            # filter_op.cuda()
            # pred_contours2d = []
            # pred_contours3d = []  
 
            for sc in [1, 2, 3]: 
                # if sc == 1:
                #     yhat_scars_ = (yhat_scars>0).float()[0]  
                # else:
                #     yhat_scars_ = (yhat_scars==sc).float()[0]  

                # # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/before.tif', yhat_scars_.cpu().numpy())
                # yhat_scars_ = resacle_z(yhat_scars_.float(), scale_factor=5.9)
                # # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/after.tif', yhat_scars_.float().cpu().numpy())

                # yhat_scars_ = filter_op(yhat_scars_[None, None].float())[0,0].detach() 
                # # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/after.tif', a.cpu().numpy()) 
                # level = (yhat_scars_.max() + yhat_scars_.min())/2
                # # level = (0.25*yhat_scars_.max() + 0.75*yhat_scars_.min()) 
                 
                # try:
                #     scar_verts_, scar_faces_, _, _ = measure.marching_cubes(yhat_scars_.cpu().numpy(), level=level.item(), allow_degenerate=False)
                # except:
                #     embed()
                # scar_verts_ = np.ascontiguousarray(scar_verts_) 
                # scar_faces_ = np.ascontiguousarray(scar_faces_)  
                # # scar_verts = torch.flip(torch.from_numpy(scar_verts_), dims=[1]) * resolution[0] 
                # scar_verts = torch.flip(torch.from_numpy(scar_verts_), dims=[1])  
                # scar_faces = torch.from_numpy(scar_faces_) 
                   
                # yhat_scars__ = resacle_z(yhat_scars_.float(), scale_factor=1/5.87)
                # yhat_scars__ = resacle(yhat_scars__, scale_factor=scale_factor, mode='bilinear') 
                # pred_voxels_rasterized =  (yhat_scars__ > 0.5).long().detach().cpu() 
                  
                # # Pred contour is in non-scaled x coordinates
                # if sc == 1:
                #     pred_contour2d = {}
                #     pred_contour3d = []
                #     for slice_id, slice in enumerate(pred_voxels_rasterized.cpu().numpy()):
                #         _, contours, _ = cv.findContours(np.uint8(slice>0), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                #         for k, ctrs in enumerate(contours):
                #             # embed()
                #             # a = np.uint8(slice>0)
                #             # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/pred_voxels.tif', 255*a)  
                #             if k == 0:
                #                 pred_contour2d[slice_id] = []
                #             ctrs = np.squeeze(ctrs)/scale_factor
                #             pred_contour2d[slice_id] += [ctrs] 

                #             if len(ctrs.shape) == 1: 
                #                 z = slice_id * np.ones(1)[:, None] 
                #                 ctrs = ctrs[None] 
                #             else:
                #                 z = slice_id * np.ones(ctrs.shape[0])[:, None] 

                #             contour_3d = np.concatenate([ctrs, z], axis=1)
                #             contour_3d = contour_3d[None] * resolution.numpy()
                #             pred_contour3d += [contour_3d]
                #             # canvas[z] = cv.drawContours(slice, [ctrs], -1, (128, 128, 0), 1) 
    
                #     pred_contours2d += [pred_contour2d]     
                #     pred_contour3d = torch.from_numpy(np.concatenate(pred_contour3d, axis=1))
                #     N = pred_contour3d.shape[1] 
                #     indices = torch.arange(N)[:, None]
                #     pred_contour3d_faces = torch.cat([indices, indices, torch.roll(indices, dims=0, shifts=[1])], dim=1)[None] 
                #     pred_contours3d += [{'vertices': pred_contour3d, 'faces':pred_contour3d_faces, 'normals':None}]     
 
                # scar_verts, scar_faces = trimesh.remesh.subdivide(scar_verts.numpy(), scar_faces.numpy())
                # scar_verts = torch.from_numpy(scar_verts)
                # scar_faces = torch.from_numpy(scar_faces) 

                # pred_scar_meshes += [{'vertices': scar_verts[None], 'faces':scar_faces[None], 'normals':None}] 
                # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/mesh.obj', scar_verts[None], scar_faces[None], None)
 
                if sc == 1:
                    y_scars_ = (data['y_scar']>0).float()[0] 
                else:
                    y_scars_ = (data['y_scar']==sc).float()[0] 
                scar_verts, scar_faces = marching_cubes(y_scars_, 0.5)  
                scar_verts = scar_verts.cpu() * resolution[0]
                scar_faces = scar_faces.cpu() 
                true_scar_meshes += [{'vertices': scar_verts[None], 'faces':scar_faces[None], 'normals':None}] 
                pred_scar_meshes += [{'vertices': scar_verts[None], 'faces':scar_faces[None], 'normals':None}] 
                # break
                
            yhat_scars = resacle(yhat_scars[0].float(), scale_factor=scale_factor)[None].detach().cpu()
            y_scars = resacle(y_scars[0].float(), scale_factor=scale_factor)[None].detach().cpu()
            y_myocardium = resacle(y_myocardium[0].float(), scale_factor=scale_factor)[None].cpu().long()
            yhat_myocardium = (pred_voxels == 2).long() 
  
            yhat_scars = yhat_scars.long()
            y_scars = y_scars.long()
 
            # -----------------------     
            # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/pred_voxels.tif', np.uint8(255 * y_scars_1.cpu().numpy()))
            # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/mesh.obj', scar_verts[None], scar_faces[None], None)
            # -----------------------    
            _, _, yhat_center_seg, yhat_center_ = pred_localizer
            y_dist = data['y_dist']
            yhat_center_seg = 0 * torch.argmax(yhat_center_seg, dim=1).cpu()
 
            y_centers = torch.amax(y_dist[0], dim=(1,2))
            y_centers = torch.logical_and(y_dist[0]==y_centers[:, None, None], y_dist[0] > 0.5)
            y_centers_coords_ = torch.nonzero(y_centers)   
            y_centers_coords = {}
            
            for p in y_centers_coords_:      
                y_centers_coords[p[0].item()] = p[1:][None]
            y_center = y_centers_coords

            y_center_seg = 0 * data['y_voxels']
            # y_center_seg[y_center_seg>1] = 1

 
            for k, v in y_center.items():
                y_center_seg[:, k, v[0, 0].long().item(), (v[0, 1].long().item()-5):(v[0, 1].long().item()+5)] = 3
                y_center_seg[:, k, (v[0, 0].long().item()-5):(v[0, 0].long().item()+5), v[0, 1].long().item()] = 3
 
            yhat_center = {}
            for k, v in enumerate(yhat_center_[0]):  
                yhat_center[k] = v[None]
                yhat_center_seg[:, k, (v[0].long().item()-5):(v[0].long().item()+5), v[1].long().item()] = 2
                yhat_center_seg[:, k, v[0].long().item(), (v[1].long().item()-5):(v[1].long().item()+5)] = 2

                if k in y_center.keys():
                    v = y_center[k].cpu()
                    yhat_center_seg[:, k, v[0, 0].long().item(), (v[0, 1].long().item()-5):(v[0, 1].long().item()+5)] = 3
                    yhat_center_seg[:, k, (v[0, 0].long().item()-5):(v[0, 0].long().item()+5), v[0, 1].long().item()] = 3

            yhat_center_seg = resacle(yhat_center_seg[0].float().cuda(), scale_factor=scale_factor)[None].long().cpu()
            y_center_seg = resacle(y_center_seg[0].float().cuda(), scale_factor=scale_factor)[None].long().cpu()
            
            # -----------------------   


            x = x.detach().data.cpu()   
            y = Structure(mesh=true_meshes, 
                            contours2d=true_contours2d,  
                            voxel=true_voxels, 
                            contours3d=true_contours3d, 
                            name=name, 
                            slices=slices, 
                            surface_points=surface_points_all, 
                            scar_mesh=true_scar_meshes,
                            scar_map=y_scars, 
                            myocardium=y_myocardium,
                            center_map=y_center_seg,
                            mu_scar=data['mu_scar'],
                            std_scar=data['std_scar'])

            y_hat = Structure(mesh=pred_meshes, 
                            contours2d=pred_contours2d, 
                            voxel=pred_voxels, 
                            contours3d=pred_contours3d,
                            scar_mesh=pred_scar_meshes,
                            myocardium=yhat_myocardium,
                            scar_map=yhat_scars,
                            center_map=yhat_center_seg,
                            mu_scar=mu_scar,
                            std_scar=std_scar) 

 
 

        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        return x, y, y_hat

    def evaluate_set(self, dataloader):
        performance = {}
        predictions = []     
        
        for i, data in enumerate(dataloader): 
            # a = run('nvidia-smi')
            # mem = int(a[1].split('\n')[9][35:40])
            # print(f'{i}:  {mem}')

            x, y, y_hat = self.predict(data, self.config)
            result = self.support.evaluate(y, y_hat, self.config) 

            predictions.append((x, y, y_hat))

            for key, value in result.items():
                if key not in performance:
                    performance[key] = []
                performance[key].append(result[key]) 
 
 
        for key, value in performance.items():
            performance[key] = np.array(performance[key])
        return performance, predictions
 
    def drawContours(self, y_voxel_z, contour, color, line_width):
        try:
            y_voxel_z = cv.drawContours(y_voxel_z, [contour], -1, color, line_width)  
        except:
            print('cv drawContours error')    
        return y_voxel_z

    def fillPoly(self, y_voxel_z, contour):
        try:
            y_voxel_z = cv.fillPoly(y_voxel_z, [contour], (1, 0, 0))  
        except:
            print('cv fillPoly error') 
        return y_voxel_z

    def scar_dist2mask(self, scarmap, mask): 
        yhat_scars = torch.zeros_like(scarmap).long() 
        # yhat_scars[scarmap<=2] = 1
        yhat_scars[scarmap>2] = 2
        yhat_scars[scarmap>5] = 3

        yhat_scarmap = mask * yhat_scars

        return yhat_scarmap

        # yhat_scarmap = yhat_scarmap.cuda()
        # yhat_scarmap_ = torch.zeros_like(yhat_scarmap) 
        
        # for k_scar in range(1, 4):
        #     k_scarmap = sample_outer_surface_in_voxel((yhat_scarmap==k_scar).long())
        #     yhat_scarmap_[k_scarmap] = k_scar
        # return yhat_scarmap_.cpu()


    def save_results_v2m(self, predictions, epoch, performence, save_path, mode, config):
  
        xs = []
        ys_voxels = [] 
        y_hats_voxels = []  
        overlays = []
        names = []

        scar_stats = []
        scarhat_stats = []
         
        stacks = []
        colors= [(0, 0, 255), (0, 255, 0), (0, 0, 192), (0, 192, 0), (0, 255, 255)]
 
        
        line_width = 2 if config.scale_factor >1 else 1
        fontScale = 1 if config.scale_factor >1 else 0.3
        color = (255, 0, 0) 
        thickness = 2 if config.scale_factor >1 else 1
        font = cv.FONT_HERSHEY_SIMPLEX  
        for i_main, data in enumerate(predictions): 
            
            x, y, y_hat = data 

            nonzero = torch.nonzero(x.sum(dim=[3,4]))[:,2]
            clip_min, clip_max = (nonzero.min(), nonzero.max()+1) 

            names += [y.name[0]]
            # embed() 

            xs_ = []
            for x_ in x[0, 0]:
                x_ = np.uint8(255*np.repeat(x_[..., None].clone().numpy(), 3, axis=2)) 
                # x_ = cv.putText(x_, y.name[0][7:], (5, 8*self.config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA)
                x_ = cv.putText(x_, y.name[0], (5, 8*config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA)

                k = 0
                for _, (key, value) in enumerate(performence.items()): 
                    if len(value.shape) == 2: 
                        # str_ = f'{key[:7]}: '+', '.join(['{:.2f}' for _ in range(config.num_classes-1)]).format(*100*value[i_main])
                        str_ = f'{key}: ' + ', '.join(map(val2str, 100*value[i_main]))
                        # x_ = cv.putText(x_, str_, (5, 16*config.scale_factor + k*8*config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA) 
                        k += 1
                xs_ += [x_[None]]
            xs_ = np.concatenate(xs_, axis=0)
            xs__ = np.copy(xs_)

            
            if y_hat.contours3d is not None: 
                save_path_i = f'{save_path}/mesh/{i_main}'
                mkdir(save_path_i) 
                for p, (true_points, pred_points) in enumerate(zip(y.contours3d, y_hat.contours3d)):
                    save_to_obj(save_path_i + '/' + mode + 'true_' + str(i_main) + '_part_' + str(p) + '_contour.obj', true_points['vertices'], true_points['faces'])
                    save_to_obj(save_path_i + '/' + mode + 'pred_' + str(i_main) + '_part_' + str(p) + '_contour.obj', pred_points['vertices'], pred_points['faces'])

            if y.mesh is not None:
                save_path_i = f'{save_path}/mesh/{i_main}'
                mkdir(save_path_i) 
                for p, true_mesh in enumerate(y.mesh):
                    save_to_obj(save_path_i + '/' + mode + 'true_' + str(i_main) + '_part_' + str(p) + '_mesh.obj', true_mesh['vertices'], true_mesh['faces'], true_mesh['normals'])
                     
            if y_hat.mesh is not None:
                save_path_i = f'{save_path}/mesh/{i_main}'
                mkdir(save_path_i) 
                for p, pred_mesh in enumerate(y_hat.mesh):
                    save_to_obj(save_path_i + '/' + mode + 'pred_' + str(i_main) + '_part_' + str(p) + '_mesh.obj', pred_mesh['vertices'], pred_mesh['faces'], pred_mesh['normals'])

            # if y.surface_points is not None:
            #     save_path_i = f'{save_path}/mesh/{i}'
            #     mkdir(save_path_i) 
            #      for p, true_surface_points in enumerate(y.surface_points):
            #         save_to_obj(save_path_i + '/' + mode + 'true_' + str(i) + '_part_' + str(p) + '_surface_points.obj', true_surface_points, None, None)
            
            if y.scar_mesh is not None: 
                for p, true_mesh in enumerate(y.scar_mesh):
                    save_to_obj(save_path_i + '/' + mode + 'true_scar_' + str(i_main) + '_region_' + str(p) + '_mesh.obj', true_mesh['vertices'], true_mesh['faces'], true_mesh['normals'])
                 
            if y_hat.scar_mesh is not None: 
                for p, pred_mesh in enumerate(y_hat.scar_mesh):
                    save_to_obj(save_path_i + '/' + mode + 'pred_scar_' + str(i_main) + '_region_' + str(p) + '_mesh.obj', pred_mesh['vertices'], pred_mesh['faces'], pred_mesh['normals'])


            y_hat_voxel = np.copy(xs_)  
            y_hat_voxels_ = []
            if y_hat.contours2d is not None:    
                for k, y_hat_ in enumerate(y_hat.contours2d):
                    y_hat_voxel_ = np.uint8(np.zeros_like(xs_[:,:,:,0]))    
                    for z, contours in y_hat_.items():
                        for contour in contours:
                            contour = np.int64(config.scale_factor * contour)  
                            y_hat_voxel[z] = self.drawContours(y_hat_voxel[z], contour, colors[k], line_width) # --------------------
                            y_hat_voxel_[z] = self.fillPoly(y_hat_voxel_[z], contour)  
                    y_hat_voxels_ += [y_hat_voxel_]
            
            y_voxel = np.copy(xs_)
            overlap = np.copy(xs_)
            if y.contours2d is not None: 
                y_voxel_ = np.uint8(np.zeros_like(xs_[:,:,:,0])) 
                overlap = np.uint8(np.zeros_like(xs_[:,:,:,0]))  
                for k, y_ in enumerate(y.contours2d):
                    for z, contour in y_.items():
                        contour = np.int64(config.scale_factor * contour.numpy())[0]
                        y_voxel[z] = self.drawContours(y_voxel[z], contour, colors[0], line_width) # --------------------
                        y_voxel_[z] = self.fillPoly(y_voxel_[z], contour) 
                    # overlap += np.uint8(np.logical_xor(y_voxel_, y_hat_voxels_[k]))
                overlap = np.uint8(255*np.repeat(overlap[..., None], 3, axis=3)) 
                overlap = blend_cpu2(torch.from_numpy(xs_).float(), torch.from_numpy(np.uint8(overlap[:,:,:,0]>0))*2, 3, factor=0.5) # *2 is hack to get the overlay the color red

                for k, y_hat_ in enumerate(y_hat.contours2d):
                    for z, contours in y_hat_.items():
                        for contour in contours:
                            contour = np.int64(config.scale_factor * contour) 
                            overlap[z] = self.drawContours(overlap[z], contour, colors[k], line_width)   
            
            
            if y_hat.scar_map is not None and y_hat.scar_map.sum() > 0:  
                 
                # y_hat_voxel = blend_cpu3(torch.from_numpy(y_hat_voxel).float(), y_hat.scar_map[0], y_hat.myocardium[0], factor=0.65) # *2 is hack to get the overlay the color red
                if config.regress_scar:    
                    yhat_scarmap = self.scar_dist2mask(y_hat.scar_map[0], y_hat.myocardium[0])  
                    y_hat_voxel = blend_cpu2(torch.from_numpy(y_hat_voxel).float(), yhat_scarmap, config.scar_num_classes, factor=0.6) # *2 is hack to get the overlay the color red
                else:
                    yhat_scarmap = y_hat.scar_map[0]
                    y_hat_voxel = blend_cpu2(torch.from_numpy(y_hat_voxel).float(), yhat_scarmap, config.scar_num_classes, factor=0.6) # *2 is hack to get the overlay the color red
 
                     
            if y.scar_map is not None and y.scar_map.sum() > 0:  
                # y_voxel = blend_cpu3(torch.from_numpy(y_voxel).float(), y.scar_map[0], y.myocardium[0], factor=0.65) # *2 is hack to get the overlay the color red
                if config.regress_scar:   
                    y_scarmap = self.scar_dist2mask(y.scar_map[0], y.myocardium[0]) 
                    y_voxel = blend_cpu2(torch.from_numpy(y_voxel).float(), y_scarmap, config.scar_num_classes, factor=0.6) # *2 is hack to get the overlay the color red
                else: 
                    y_scarmap = y.scar_map[0]
                    y_voxel = blend_cpu2(torch.from_numpy(y_voxel).float(), y_scarmap, config.scar_num_classes, factor=0.6) # *2 is hack to get the overlay the color red
             
            # if y_hat.center_map is not None and y_hat.center_map.sum() > 0:    
            #     # overlap = blend_cpu2(torch.from_numpy(overlap).float(), y_hat.center_map[0], 4, factor=0.3) # *2 is hack to get the overlay the color red
            #     y_hat_voxel = blend_cpu(torch.from_numpy(y_hat_voxel).float(), y_hat.center_map[0], 4, factor=0) # *2 is hack to get the overlay the color red
            # if y.center_map is not None and y.center_map.sum() > 0:   
            #     # overlap = blend_cpu2(torch.from_numpy(overlap).float(), y.center_map[0], 4, factor=0.3) # *2 is hack to get the overlay the color red
            #     y_voxel = blend_cpu(torch.from_numpy(y_voxel).float(), y.center_map[0], 4, factor=0) # *2 is hack to get the overlay the color red
            # embed()
            xs_ = []
            for p, x_ in enumerate(y_voxel):  
                # x_ = cv.putText(x_, 'Image', (260, 620) , font, fontScale, color, thickness, cv.LINE_AA)
                x_ = cv.putText(x_, 'Image + Ground truth', (140, 620) , font, fontScale, color, thickness, cv.LINE_AA)
                value = performence['y_N']
                th1, th2, lv = value[i_main][p]/(config.scale_factor**2)  
                x_ = cv.putText(x_, f'Th1<Voxels<=Th2: {th1}[N], {th1 * 1 * 1 * 6 / 1000:.2f}[ml]', (5, 16*config.scale_factor + 0*8*config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA) 
                x_ = cv.putText(x_, f'Voxels>Th2: {th2}[N], {th2 * 1 * 1 * 6 / 1000:.2f}[ml]', (5, 16*config.scale_factor + 1*8*config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA) 
                x_ = cv.putText(x_, f'LV: {lv}[N], {lv * 1 * 1 * 6 / 1000:.2f}[ml] ', (5, 16*config.scale_factor + 2*8*config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA) 
                 
                xs_ += [x_[None]]
            y_voxel = np.concatenate(xs_, axis=0)
                
            xs_ = []
            for p, x_ in enumerate(y_hat_voxel):  
                x_ = cv.putText(x_, 'Image + HEARTS output', (140, 620) , font, fontScale, color, thickness, cv.LINE_AA)
                value = performence['yhat_N']
                yhatth1, yhatth2, yhatlv = value[i_main][p]/(config.scale_factor**2)
                # yth1, yth2, ylv = performence['y_N'][i_main][p]/(config.scale_factor**2)  
                x_ = cv.putText(x_, f'Th1<Voxels<=Th2: {yhatth1}[N], {yhatth1 * 1 * 1 * 6 / 1000:.2f}[ml]', (5, 16*config.scale_factor + 0*8*config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA) 
                x_ = cv.putText(x_, f'Voxels>Th2: {yhatth2}[N], {yhatth2 * 1 * 1 * 6 / 1000:.2f}[ml]', (5, 16*config.scale_factor + 1*8*config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA) 
                x_ = cv.putText(x_, f'LV: {yhatlv}[N], {yhatlv * 1 * 1 * 6 / 1000:.2f}[ml]', (5, 16*config.scale_factor + 2*8*config.scale_factor) , font, fontScale, color, thickness, cv.LINE_AA) 
                 
                xs_ += [x_[None]]
            y_hat_voxel = np.concatenate(xs_, axis=0)
             
            xs.append(xs__[clip_min:clip_max])
            ys_voxels.append(y_voxel[clip_min:clip_max])
            y_hats_voxels.append(y_hat_voxel[clip_min:clip_max])
            overlays.append(overlap[clip_min:clip_max])
               
            if y.slices is not None: 
                stacks += [y.slices]
                # for slice_id, slice in y.slices.items(): 
                #     file_name = f'slice_{slice_id}'   
                #     save_path_i = f'{save_path}/mesh/{i_main}'   

                #     # print(save_path_i)
                #     # print(slice_id)
                #     # save_to_texture_obj(save_path_i, file_name, slice['slice_vertices'], slice['slice_faces'], slice['slice_uv'], slice['image']) 
                #     img = torch.from_numpy(y_hat_voxel[slice_id]).float()  
                #     img = torch.flip(img, dims=[0])
                #     save_to_texture_obj(save_path_i, file_name, slice['slice_vertices'], slice['slice_faces'], slice['slice_uv'], img)
            
        xs = np.concatenate(xs, axis=0)
        overlays = np.concatenate(overlays, axis=0)
        ys_voxels = np.concatenate(ys_voxels, axis=0) 
        y_hats_voxels = np.concatenate(y_hats_voxels, axis=0) 

        overlay = np.concatenate([xs, ys_voxels, y_hats_voxels], axis=2)
        io.imsave(save_path + mode + 'overlay_y_hat.tif', overlay) 
      
        if performence is not None:
            for key in ['jaccard_lv', 'jaccard_scar']:
            # for key, value in performence.items(): 
                performence_mean = np.mean(performence[key], axis=0) 
                summary = f'{key}-{epoch}: ' + ', '.join(map(val2str, 100*performence_mean)) + ' | mu: ' + val2str(100*np.mean(performence_mean))
                append_line(save_path + mode + 'summary' + key + '.txt', summary) 
                print(summary)
                # all_results = [('{}: ' + ', '.join(['{:.8f}' for _ in range(config.num_classes-1)])).format(*((i+1,) + tuple(vals))) for i, vals in enumerate(performence[key])]
                # print(all_results)
                    
            
            write_lines(save_path + mode + 'scar_volume' + '.txt', '')  
            y_N = performence['y_N']
            yhat_N = performence['yhat_N']

            scar_error = []
            f = open(save_path + mode + 'scar_volume_per_slice.csv', 'w')
            writer = csv.writer(f)
            writer.writerow(['file name', 
                            'LV[ml] - Gt', 
                            'LV[ml] - HeARTS', 
                            'Th1 < Voxels <= Th2[ml] - Gt', 
                            'Th1 < Voxels <= Th2[ml] - HeARTS',
                            'Volume > Thr2[ml] - Gt', 
                            'Volume > Thr2[ml] - HeARTS'])
            f2 = open(save_path + mode + 'scar_volume_total.csv', 'w')
            writer2 = csv.writer(f2)
            writer2.writerow(['file name', 
                            'LV[ml] - Gt', 
                            'LV[ml] - HeARTS', 
                            'Th1 < Voxels <= Th2[ml] - Gt', 
                            'Th1 < Voxels <= Th2[ml] - HeARTS',
                            'Volume > Thr2[ml] - Gt', 
                            'Volume > Thr2[ml] - HeARTS'])

            for fileID, (y_N_i, yhat_N_i) in enumerate(zip(y_N, yhat_N)):
                name = names[fileID] 
                
                y_N_volume = 0
                yhat_N_volume = 0
                for y_N_ij, yhat_N_ij in zip(y_N_i, yhat_N_i):
                    if y_N_ij.sum() > 0:
                        y_N_ij = y_N_ij/(config.scale_factor**2)
                        yhat_N_ij = yhat_N_ij/(config.scale_factor**2)

                        y_N_ij_volume = y_N_ij * 1 * 1 * 6 / 1000
                        yhat_N_ij_volume = yhat_N_ij * 1 * 1 * 6 / 1000

                        y_N_volume += y_N_ij_volume
                        yhat_N_volume += yhat_N_ij_volume 
                        writer.writerow([name, 
                                        f'{y_N_ij_volume[2]:.2f}', 
                                        f'{yhat_N_ij_volume[2]:.2f}', 
                                        f'{y_N_ij_volume[0]:.2f}', 
                                        f'{yhat_N_ij_volume[0]:.2f}', 
                                        f'{y_N_ij_volume[1]:.2f}', 
                                        f'{yhat_N_ij_volume[1]:.2f}'])

                        values = ', '.join(map(val2str, y_N_ij)) + ', ' + ', '.join(map(val2str, yhat_N_ij)) 
                        append_line(save_path + mode + 'scar_volume_per_slice' + '.txt',values)

                        error = np.abs(y_N_ij - yhat_N_ij)
                        scar_error += [error[None]]
                writer.writerow(['']) 
                writer2.writerow([name, 
                                f'{y_N_volume[2]:.2f}', 
                                f'{yhat_N_volume[2]:.2f}', 
                                f'{y_N_volume[0]:.2f}', 
                                f'{yhat_N_volume[0]:.2f}', 
                                f'{y_N_volume[1]:.2f}', 
                                f'{yhat_N_volume[1]:.2f}'])
            f.close() 
            f2.close() 
            scar_error = np.concatenate(scar_error, axis=0)
            mean_scar_error = np.mean(scar_error,axis=0)
            std_scar_error = np.std(scar_error,axis=0)
            summary = f'scar-[error-mean]-{epoch}: ' + ', '.join(map(val2str, mean_scar_error)) + ' | mu: ' + val2str(np.mean(mean_scar_error))     
            print(summary)  
            summary = f'scar[error-std]-{epoch}: ' + ', '.join(map(val2str, std_scar_error)) + ' | mu: ' + val2str(np.mean(std_scar_error))     
            print(summary)


            # # embed() 
            # yscar_myocardium = y_scars > 0  

            # surf1 = Meshes(verts=list(pred_meshes[0]['vertices']), faces=list(pred_meshes[0]['faces'])) # inner
            # surf2 = Meshes(verts=list(pred_meshes[1]['vertices']), faces=list(pred_meshes[1]['faces'])) # outer
            # surf_points = [sample_points_from_meshes(surf2, 6000), sample_points_from_meshes(surf1, 6000)] 

            # pred_contour3d = {}
            # pred_contour3d[0] = []
            # pred_contour3d[1] = []
            # scenters = {}
            # scenters[0] = []
            # scenters[1] = []   

            # shifts = {} 
            # k = 0 # k = 0 outer, k = 1 inner
            # gap = 6
            # for slice_id, slice in enumerate(yscar_myocardium[0].numpy()):
            #     _, contours, _ = cv.findContours(np.uint8(slice>0), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            #     # for k, ctrs in enumerate(contours):  
            #     if len(contours) >= k+1:
            #         ctrs = contours[k]
            #         # Center from scarnet 
            #         ctrs = np.squeeze(ctrs)/scale_factor 

            #         if len(ctrs.shape) == 1: 
            #             z = slice_id * np.ones(1)[:, None] 
            #             ctrs = ctrs[None] 
            #         else:
            #             z = slice_id * np.ones(ctrs.shape[0])[:, None] 

            #         contour_3d = np.concatenate([ctrs, z], axis=1)
            #         contour_3d = contour_3d[None] * resolution.numpy() 
            #         scar_center = np.mean(contour_3d, axis=1)[None]
                    
            #         # Center from V2M
            #         surf_z_levels = surf_points[k][0, :, 2]
            #         dist = np.abs(surf_z_levels - np.unique(contour_3d[0,:,2]))
            #         points_on_level = surf_points[k][:, dist < gap] 
            #         v2m_center = torch.mean(points_on_level, dim=1)[None].numpy() 

            #         shifts[slice_id] = (v2m_center[:,:,:2] - scar_center[:,:,:2]) * scale_factor

            #         scenters[k] += [scar_center]
            #         pred_contour3d[k] += [contour_3d]

            # # pred_contour3d[0] = np.concatenate(pred_contour3d[0], axis=1)
            # # pred_contour3d[1] = np.concatenate(pred_contour3d[1], axis=1)
            # # scenters[0] = np.concatenate(scenters[0], axis=1)
            # # scenters[1] = np.concatenate(scenters[1], axis=1)
            # # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/contour3d1.obj', torch.from_numpy(pred_contour3d[0]), None, None)
            # # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/contour3d2.obj', torch.from_numpy(pred_contour3d[1]), None, None)

            # # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/centers3d1.obj', torch.from_numpy(scenters[0]), None, None)
            # # save_to_obj('/cvlabdata2/cvlab/datasets_udaranga/outputs/centers3d2.obj', torch.from_numpy(scenters[1]), None, None)
       
            # print('shifted')
            # embed()
            # for slice_id in shifts.keys():
            #         x[0, 0, slice_id] = shift_2d_replace(data=x[0, 0, slice_id],
            #                                             dx=round(shifts[slice_id][0, 1].detach().item()),
            #                                             dy=round(shifts[slice_id][0, 0].detach().item()))

            #         true_voxels[0, slice_id] = shift_2d_replace(data=true_voxels[0, slice_id],
            #                                             dx=round(shifts[slice_id][0, 1].detach().item()),
            #                                             dy=round(shifts[slice_id][0, 0].detach().item()))

            #         y_scars[0, slice_id] = shift_2d_replace(data=y_scars[0, slice_id],
            #                                             dx=round(shifts[slice_id][0, 1].detach().item()),
            #                                             dy=round(shifts[slice_id][0, 0].detach().item()))

            #         yhat_scars[0, slice_id] = shift_2d_replace(data=yhat_scars[0, slice_id],
            #                                             dx=round(shifts[slice_id][0, 1].detach().item()),
            #                                             dy=round(shifts[slice_id][0, 0].detach().item()))

            #         y_myocardium[0, slice_id] = shift_2d_replace(data=y_myocardium[0, slice_id],
            #                                             dx=round(shifts[slice_id][0, 1].detach().item()),
            #                                             dy=round(shifts[slice_id][0, 0].detach().item()))

            #         yhat_myocardium[0, slice_id] = shift_2d_replace(yhat_myocardium=y_dist[0, slice_id],
            #                                             dx=round(shifts[slice_id][0, 1].detach().item()),
            #                                             dy=round(shifts[slice_id][0, 0].detach().item())) 