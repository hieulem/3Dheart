
import os
import logging

import numpy as np
#import cv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from functools import reduce
import cv2
import sys
from IPython import embed
import copy
import yaml
from easydict import EasyDict

import matplotlib as mpl


volume_suffix = '' 

class Modes:
    TRAINING = 'training'
    VALIDATION = 'validation'
    TESTING = 'testing'
    DEPLOY = 'deploy'
    ALL = 'all'
    def __init__(self): 
        dataset_splits = [Modes.TRAINING, Modes.VALIDATION, Modes.TESTING, Modes.DEPLOY]


def write_lines(path, lines):
    f = open(path, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
 
def append_line(path, line):
    f = open(path, 'a')
    f.write(line + '\n')
    f.close() 

def pytorch_count_params(model):
  "count number trainable parameters in a pytorch model"
  total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in model.parameters())
  return total_params

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def blend(img, mask):

    img = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2RGB)

    rows, cols, d = img.shape
    pre_synaptic = np.zeros((rows, cols, 1))
    pre_synaptic[mask == 1] = 1

    synpase = np.zeros((rows, cols, 1))
    synpase[mask == 2] = 1

    post_synaptic = np.zeros((rows, cols, 1))
    post_synaptic[mask == 3] = 1

    color_mask = np.dstack((synpase, pre_synaptic , post_synaptic))
    color_mask = np.uint8(color_mask*255)

    blended = cv2.addWeighted(img, 0.8, color_mask, 0.2, 0)
    return blended
 

def crop_slices(shape1, shape2):
    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2) for sh1, sh2 in zip(shape1, shape2)]
    return slices

def crop_and_merge(tensor1, tensor2):

    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)

    return torch.cat((tensor1[slices], tensor2), 1)


def _box_in_bounds(box, image_shape):
    newbox = []
    pad_width = []

    for box_i, shape_i in zip(box, image_shape):
        pad_width_i = (max(0, -box_i[0]), max(0, box_i[1] - shape_i))
        newbox_i = (max(0, box_i[0]), min(shape_i, box_i[1]))

        newbox.append(newbox_i)
        pad_width.append(pad_width_i)

    needs_padding = any(i != (0, 0) for i in pad_width)

    return newbox, pad_width, needs_padding

def crop_indices(image_shape, patch_shape, center):
    box = [(i - ps // 2, i - ps // 2 + ps) for i, ps in zip(center, patch_shape)]
    box, pad_width, needs_padding = _box_in_bounds(box, image_shape)
    slices = tuple(slice(i[0], i[1]) for i in box)
    return slices, pad_width, needs_padding

def crop(image, patch_shape, center, mode='constant'):
    # embed()
    slices, pad_width, needs_padding = crop_indices(image.shape, patch_shape, center)
    patch = image[slices]

    if needs_padding and mode != 'nopadding':
        if isinstance(image, np.ndarray):
            if len(pad_width) < patch.ndim:
                pad_width.append((0, 0))
            patch = np.pad(patch, pad_width, mode=mode)
        elif isinstance(image, torch.Tensor):
            assert len(pad_width) == patch.dim(), "not supported"
            # [int(element) for element in np.flip(np.array(pad_width).flatten())]
            patch = F.pad(patch, tuple([int(element) for element in np.flip(np.array(pad_width), axis=0).flatten()]), mode=mode)

    return patch

def contour_shift(contours, shift):  
    # -----------------------
    # contours data structure
    # -----------------------
    # contours keys: classes of contours, eg: `1,2,9,10`, inner wall, outer wall, scar1, scar2
    # contour = contours[key]
    # contour keys: slice indices, eg: `1,2,3,4,5,6,7,8,9`
    # contour_2d, contour_3d = slice_contour <= contour[key] 
    # -----------------------
    
    contours = copy.deepcopy(contours)
    for class_id, contour in contours.items():
        for slice_id, slice_contour in contour.items(): 
            if len(slice_contour.shape) == 3:
                contours[class_id][slice_id] = slice_contour - shift[None]
            else: 
                contours[class_id][slice_id] = slice_contour - shift # contour_2d, contour_3d = slice_contour 
    return contours

def contour_pad(contours, pad_width):
    # -----------------------
    # contours data structure
    # -----------------------
    # contours keys: classes of contours, eg: `1,2,9,10`, inner wall, outer wall, scar1, scar2
    # contour = contours[key]
    # contour keys: slice indices, eg: `1,2,3,4,5,6,7,8,9`
    # contour_2d, contour_3d = slice_contour <= contour[key] 
    # -----------------------

    contours_padded = {} 

    # Padding shifts 
    z_shift = pad_width[-3][0].item()
    xy_shift = torch.tensor([pad_width[-1][0], pad_width[-2][0]])[None]
    
    for class_id, contour in contours.items():  
        counter_padded = {}
        for slice_id, slice_contour in contour.items():    
            if len(slice_contour.shape) == 3: 
                counter_padded[slice_id + z_shift] = slice_contour + xy_shift[None]
            else:
                counter_padded[slice_id + z_shift] = slice_contour + xy_shift # contour_2d, contour_3d = slice_contour 
        contours_padded[class_id] = counter_padded
    return contours_padded

def permute(images, contours):
    images_permuted = []
    for x in images:
        images_permuted += [x.permute([0, 2, 1])] 
     
    contours = copy.deepcopy(contours)
    for class_id, contour in contours.items():
        for slice_id, slice_contour in contour.items():   
            contours[class_id][slice_id] = torch.flip(slice_contour, [1]) # contour_2d, contour_3d = slice_contour     
    return images_permuted, contours

def flip(images, dims, contours):
    images_permuted = []
    
    for x in images:
        images_permuted += [torch.flip(x, dims=dims)] 
        shape = x.shape
 
    if 0 in dims:
        contours_flipped = {} 
        D, _, _ = shape
        for class_id, contour in contours.items():  
            counter_padded = {}
            for slice_id, slice_contour in contour.items():    
                counter_padded[D-slice_id-1] = slice_contour # contour_2d, contour_3d = slice_contour , -1 since arrays starts from 0
            contours_flipped[class_id] = counter_padded
        contours = contours_flipped
 
    _, H, W = shape 
    for class_id, contour in contours.items():
        for slice_id, contour_2d in contour.items():  
            if 1 in dims:
                contour_2d[:, 1] = H - contour_2d[:, 1]   
            if 2 in dims:
                contour_2d[:, 0] = W - contour_2d[:, 0]
            contours[class_id][slice_id] = contour_2d # contour_2d, contour_3d = slice_contour 
    return images_permuted, contours

def crop_images_and_contours(images, contours, patch_shape, center, mode='constant'):
 
    # image = temp.clone()
    # contours = copy.deepcopy(temp_cont)
    # center = copy.deepcopy(temp_center)
    # patch_shape = tuple((32, 128, 128))  

    slices, pad_width, needs_padding = crop_indices(images[0].shape, patch_shape, center)
    
    patches = []
    for image in images:
        patches += [image[slices]]
 
    # shift = torch.tensor([slices[2].start, slices[1].start])[None] 
    shift = torch.tensor([slices[-1].start, slices[-2].start])[None] 
    contours = contour_shift(contours, shift) 
    
    if needs_padding and mode is not 'nopadding':
        contours = contour_pad(contours, pad_width)
        for i, patch in enumerate(patches):
            if isinstance(patch, np.ndarray):  
                assert len(pad_width) == patch.ndim, "not supported" 
                patches[i] = np.pad(patch, pad_width, mode=mode) 
            elif isinstance(patch, torch.Tensor): 
                if len(pad_width) != patch.dim():
                    embed()
                assert len(pad_width) == patch.dim(), "not supported" 
                patches[i] = F.pad(patch, tuple([int(element) for element in np.flip(np.array(pad_width), axis=0).flatten()]), mode=mode)

    # -- for debugging --
    # from skimage import io
    # import cv2 as cv
  
    # for k, contour in contours.items():
    #     for p, v in contour.items(): 
    #         contour_2d, _ = v
    #         patch[p] = torch.from_numpy(cv.drawContours(patch[p].cpu().numpy(), [np.int64((contour_2d).cpu().numpy())], -1, (255, 255, 0), 1))  
    # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/overlay1_{}.tif'.format(0), patch.numpy())
 
    return patches, contours

def hist_normalize(x):
    D, H, W = x.shape
    x1d = np.reshape(x, (D*H,W))
    x1d = np.uint8(255*(x1d-x.min())/(x.max()-x.min()))
    x = np.reshape(cv2.equalizeHist(x1d), (D, H, W))
    return x

# def clip(x):
#     # clipping
#     x = x.numpy()
#     x_min = np.percentile(x, 5)
#     x_max = np.percentile(x, 95)
#     x[x<x_min] = x_min
#     x[x>x_max] = x_max
#     x = np.uint8(255*(x-x.min())/(x.max()-x.min()))
#     x = torch.from_numpy(x).cuda().float()
#     x = x.cuda()

def blend(img, labels, num_classes):
    colors = torch.tensor([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255]]).cuda().float()


    img = img[..., None].repeat(1, 1, 1, 3)
    masks = torch.zeros_like(img)
    for cls in range(1, num_classes):
        masks += torch.ones_like(img) * colors[cls] * (labels == cls).float()[:, :, :, None]

    overlay = np.uint8((255 * img * 0.8 + masks * 0.2).data.cpu().numpy())
    return overlay

def blend_cpu2(img, labels, num_classes, factor=0.8):
    colors = torch.tensor([[0, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 255]]).float()
    # colors = torch.tensor([[0, 0, 0], [255, 0, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255]]).float()

    # img = img[..., None].repeat(1, 1, 1, 3)
    masks = torch.zeros_like(img)

    for cls in range(1, num_classes):
        masks += torch.ones_like(img) * colors[cls] * (labels == cls).float()[:, :, :, None]
  
    overlay = img.clone() 
    masks_ = np.repeat(masks.sum(dim=3)[..., None],3, axis=3)
    overlay[masks_>0] = img[masks_>0] * factor + masks[masks_>0] * (1-factor) 
    overlay = np.uint8(overlay.data.numpy()) 
    return overlay

def blend_cpu3(img, labels, mask, factor=0.8):
    # embed()
    sigma_max = 8
    mask_ = mask[..., None].repeat(1,1,1,3)
    cmap = mpl.colormaps['seismic']
    labels_ = np.clip(labels.cpu().detach().numpy(), -sigma_max, sigma_max)/sigma_max
    labels_ = cmap(labels_)
    labels_ = 255 * labels_[:,:,:,:3]


    overlay = img.clone().numpy() 
    overlay[mask_>0] = img[mask_>0] * factor + labels_[mask_>0] * (1-factor) 
    # overlay = img.cpu().numpy() * factor + 255 * labels_[:,:,:,:3] * (1-factor) 

    # # img = img[..., None].repeat(1, 1, 1, 3)
    # masks = torch.zeros_like(img)
    # for cls in range(1, num_classes):
    #     masks += torch.ones_like(img) * colors[cls] * (labels == cls).float()[:, :, :, None]
  
    # overlay = img.clone() 
    # masks_ = np.repeat(masks.sum(dim=3)[..., None],3, axis=3)
    # overlay[masks_>0] = img[masks_>0] * factor + masks[masks_>0] * (1-factor) 
    overlay = np.uint8(overlay) 
    return overlay

def blend_cpu(img, labels, num_classes, factor=0.8):
    # colors = torch.tensor([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255]]).float()
    colors = torch.tensor([[0, 0, 0], [0, 0, 0], [255, 255, 0], [255, 0, 0], [255, 0, 255]]).float()
    
    img = img.clone()
    # img = 255 * img[..., None].repeat(1, 1, 1, 3) 
    # embed()
    masks = torch.zeros_like(img)
    for cls in range(1, num_classes):
        mask = (labels == cls) 
        img[mask] = colors[cls]

    img = np.uint8(img)
    return img


def load_yaml(path):
    """
    loads a YAML file
    Args:
        path: (string) path to the configuration.yaml file to load
    :return:
        config: config file processed into a dictionary by EasyDict
    """
    file = yaml.load(open(path), Loader=yaml.FullLoader)
    config = EasyDict(file)

    return config

def val2str(val):
    return f'{val:.2f}'

def gaussian_filter(filter_dim, sigma):
    D, H, W = filter_dim
    base_grid = torch.zeros((1, D, H, W, 3))

    w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
    h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
    d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

    base_grid[:, :, :, :, 0] = w_points
    base_grid[:, :, :, :, 1] = h_points
    base_grid[:, :, :, :, 2] = d_points 
    base_grid = base_grid[0]

    filter = torch.exp(-torch.sum(base_grid**2, dim=3)/(2*sigma**2))
    filter = filter/filter.sum()
    return filter