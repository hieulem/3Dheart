
from IPython import embed

import torch
from utils.rasterize.rasterize2 import rasterize_vol
from pytorch3d.structures import Meshes
from utils.utils_common import append_line, write_lines
from utils.utils_voxel2mesh.file_handle import save_to_obj
import numpy as np
from skimage import io 
class Structure(object):

    def __init__(self, 
                voxel=None, 
                mesh=None,   
                name=None):
        self.voxel = voxel 
        self.mesh = mesh    
        self.name = name 

# def blend_cpu(img, labels, num_classes, factor=0.8):
#     # colors = torch.tensor([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255]]).float()
#     colors= [(0, 0, 255), (0, 255, 0), (255, 0, 0), (127, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255) ]

#     img = img.clone()
#     # img = np.uint8(255 * img[..., None].repeat(1, 1, 1, 3))
#     # embed() 
#     for cls in range(1, num_classes):
#         mask = (labels == cls) 
#         img[mask] = colors[cls]

#     # img = np.uint8(img)
#     return img


def blend_cpu(img, labels, num_classes, factor=0.8):
    colors= [(0, 0, 255), (0, 255, 0), (255, 0, 0), (127, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255) ]
 
    # img = img[..., None].repeat(1, 1, 1, 3)
    masks = torch.zeros_like(img) 
    for cls in range(1, num_classes):
        masks += torch.ones_like(img) * torch.tensor(colors[cls]) * (labels == cls).float()[:, :, :, None]
   
    overlay = img.clone() 
    masks_ = np.repeat(masks.sum(dim=3)[..., None],3, axis=3)
    overlay[masks_>0] = img[masks_>0] * factor + masks[masks_>0] * (1-factor) 
    overlay = np.uint8(overlay.data.numpy()) 
    return overlay

def predict(model, data, config):
     

    pred, data = model(data) 
    x = data['x']
    true_voxels = data['y_voxels']
    name = data['name']

    pred_voxels_rasterized_all = [] 
    pred_meshes = []
    true_meshes = []

    shape = torch.tensor(x.shape)  
    shape = shape[2:].flip([0])
    pred_voxels = torch.zeros_like(x)[:,0].long()
    for c in range(config.num_classes-1):  
        # embed()

        pred_vertices = pred[c][-1][0].detach().data.cpu()
        pred_faces = pred[c][-1][1].detach().data.cpu()
        pred_probs = pred[c][-1][3].detach().data.cpu()   
        pred_meshes += [{'vertices': (pred_vertices/2 + 0.5) * (shape-1) , 'faces':pred_faces, 'normals':None}] 
        
        
        if len(data['vertices_mc']) > 0:
            true_vertices = data['vertices_mc'][c].data.cpu()  
            true_faces = data['faces_mc'][c].data.cpu()    
            true_meshes += [{'vertices': (true_vertices/2 + 0.5) * (shape-1), 'faces':true_faces, 'normals':None}]  
        else:
            true_vertices = true_faces = true_meshes = None
            # true_meshes += [{'vertices': None, 'faces':None, 'normals':None}]  
            # true_contours3d += [{'vertices': None, 'faces':None, 'normals':None}]  
        


        verts = torch.flip(pred_vertices, [2]) 
        faces = pred_faces  
        mesh = Meshes(verts=verts.detach(), faces=faces.detach().long()) 
        pred_voxels_rasterized = rasterize_vol(mesh, x.shape[2:])   
        pred_voxels[pred_voxels_rasterized[None]>0] = c + 1 

    

    x = x.detach().data.cpu()   
    y = Structure(mesh=true_meshes,  
                    voxel=true_voxels,  
                    name=name)

    y_hat = Structure(mesh=pred_meshes,  
                    voxel=pred_voxels) 

    return x, y, y_hat

def save_results_full_hearts(predictions, epoch, performence, save_path, mode, config): 
    
    xs = []
    ys_voxels = []  
    y_hats_voxels = [] 

    for i, data in enumerate(predictions):
        x, y, y_hat = data

        xs.append(x[0, 0])
  
        for p, (true_mesh, pred_mesh) in enumerate(zip(y.mesh, y_hat.mesh)):
            save_to_obj(save_path + '/mesh/' + mode + 'true_' + str(i) + '_part_' + str(p) + '.obj', true_mesh['vertices'], true_mesh['faces'], true_mesh['normals'])
            save_to_obj(save_path + '/mesh/' + mode + 'pred_' + str(i) + '_part_' + str(p) + '.obj', pred_mesh['vertices'], pred_mesh['faces'], pred_mesh['normals'])

        ys_voxels.append(y.voxel[0]) 
        y_hats_voxels.append(y_hat.voxel[0])  
 
 
    
    if performence is not None:
        for key, value in performence.items():
            performence_mean = np.mean(performence[key], axis=0)
            summary = ('{}: ' + ', '.join(['{:.8f}' for _ in range(config.num_classes-1)])).format(epoch, *performence_mean)
            append_line(save_path + mode + 'summary' + key + '.txt', summary)
            print(('{} {}: ' + ', '.join(['{:.8f}' for _ in range(config.num_classes-1)])).format(epoch, key, *performence_mean))

            # all_results = [('{}: ' + ', '.join(['{:.8f}' for _ in range(config.num_classes-1)])).format(*((i+1,) + tuple(vals))) for i, vals in enumerate(performence[key])]
            # write_lines(save_path + mode + 'all_results_' + key + '.txt', all_results)

        
    xs = torch.cat(xs, dim=0).cpu() 
    xs = np.uint8(255 * xs[..., None].repeat(1, 1, 1, 3))
    ys_voxels = torch.cat(ys_voxels, dim=0).cpu()
    y_hats_voxels = torch.cat(y_hats_voxels, dim=0).cpu()
 
    overlay_y_hat = blend_cpu(torch.from_numpy(xs).float(), torch.from_numpy(np.uint8(y_hats_voxels)), config.num_classes)
    overlay_y = blend_cpu(torch.from_numpy(xs).float(), ys_voxels, config.num_classes) 
 
    xs = np.uint8(xs)
    overlay = np.concatenate([xs, overlay_y, overlay_y_hat], axis=2)
    io.imsave(save_path + mode + 'overlay_y_hat.tif', overlay)

