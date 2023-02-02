import numpy as np
import torch
  
import time
from IPython import embed
from scipy.io import savemat
from utils.utils_common import mkdir 
from skimage import io 

def read_obj(filepath):
    vertices = []
    faces = [] 
    normals = []   
    with open(filepath) as fp:
        line = fp.readline() 
        cnt = 1 
        while line: 
            if line[0] is not '#': 
                cnt = cnt + 1 
                values = [float(x) for x in line.split('\n')[0].split(' ')[1:]] 
                if line[:2] == 'vn':  
                    normals.append(values)
                elif line[0] == 'v':
                    vertices.append(values)
                elif line[0] == 'f':
                    faces.append(values) 
            line = fp.readline()
        vertices = np.array(vertices)
        normals = np.array(normals)
        faces = np.array(faces)
        faces = np.int64(faces) - 1
        if len(normals) > 0:
            return vertices, faces, normals
        else:
            return vertices, faces


def save_to_obj(filepath, points, faces, normals=None, texture=None): 
    with open(filepath, 'w') as file:
        vals = ''  
        for i, point in enumerate(points[0]):
            point = point.data.cpu().numpy()
            vals += 'v ' + ' '.join([str(val) for val in point]) + '\n'
        if normals is not None:
            for i, normal in enumerate(normals[0]):
                normal = normal.data.cpu().numpy()
                vals += 'vn ' + ' '.join([str(val) for val in normal]) + '\n'
        if texture is not None:
            for i, t in enumerate(texture[0]):
                t = t.data.cpu().numpy()
                vals += 'vt ' + ' '.join([str(val) for val in t]) + '\n'

        if faces is not None and len(faces) > 0: 
            for i, face in enumerate(faces[0]):
                face = face.data.cpu().numpy()
                vals += 'f ' + ' '.join([str(val+1) for val in face]) + '\n'

        file.write(vals)

def save_to_texture_obj(root, file_name, points, faces, uv, image): 

    mkdir(f'{root}/{file_name}')
    with open(f'{root}/{file_name}/{file_name}.obj', 'w') as file:
        vals = f'mtllib {file_name}.mtl\n'  
        for i, point in enumerate(points[0]):
            point = point.data.cpu().numpy()
            vals += 'v ' + ' '.join([str(val) for val in point]) + '\n' 
        if uv is not None:
            for i, t in enumerate(uv[0]):
                t = t.data.cpu().numpy()
                vals += 'vt ' + ' '.join([str(val) for val in t]) + '\n'

        if faces is not None and len(faces) > 0:
            for i, face in enumerate(faces[0]):
                face = face.data.cpu().numpy()
                vals += 'f ' + ' '.join([f'{str(val+1)}/{str(val+1)}' for val in face]) + '\n'
        file.write(vals)
    with open(f'{root}/{file_name}/{file_name}.mtl', 'w') as file:
        vals = ['newmtl material0\n',
                'Ka 1.000000 1.000000 1.000000\n',
                'Kd 1.000000 1.000000 1.000000\n',
                'Ks 0.000000 0.000000 0.000000\n',
                'Tr 1.000000\n',
                'illum 1\n',
                'Ns 0.000000\n',
                f'map_Kd ./{file_name}.jpg']
        file.write(''.join(vals))

    image = image.cpu().numpy()
    image = 255*(image - image.min())/(image.max()-image.min())
    io.imsave(f'{root}/{file_name}/{file_name}.jpg', np.uint8(image))
        

def get_slice_mesh(x_original, scale_factor):
    _, _, D, H, W = x_original.shape
    x_ = torch.linspace(0, W-1, steps=W)
    y_ = torch.linspace(0, H-1, steps=H) 

    u_ = x_/(W-1)
    v_ = y_/(H-1)
 

    grid_x, grid_y = torch.meshgrid(x_, y_, indexing='ij') 
    grid_u, grid_v = torch.meshgrid(u_, v_, indexing='ij') 
    grid = torch.cat([grid_x[:,:,None], grid_y[:,:,None]], dim=2).reshape(-1,2) /scale_factor  
    grid_uv = torch.cat([grid_u[:,:,None], grid_v[:,:,None]], dim=2).reshape(-1,2)[None]


    ids = torch.arange(H*W).reshape(H, W)

    f1 = ids[:-1,:-1][..., None]
    f2 = ids[:-1,1:][..., None]
    f3 = ids[1:,1:][..., None]
    f = [torch.cat([f3,f2,f1], dim=2).reshape(-1,3)]

    f1 = ids[:-1,:-1][..., None]
    f2 = ids[1:,:-1][..., None]
    f3 = ids[1:,1:][..., None]
    f += [torch.cat([f2,f3,f1], dim=2).reshape(-1,3)]

    f = torch.cat(f, dim=0)[None]

    return grid, grid_uv, f
