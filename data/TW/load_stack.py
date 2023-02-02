from re import S
import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET
from os import listdir
from pydicom.filereader import dcmread
from IPython import embed 
import torch.nn.functional as F
from scipy.ndimage import shift 
from pydicom.fileset import FileSet
from utils.utils_common import crop

def stringtoarray(txt):
        array = txt.split() 
        array = np.array([float(c) for c in array])
        return array 

def clean_border_pixels(image, gap):
    '''
    :param image:
    :param gap:
    :return:
    '''
    assert len(image.shape) == 3, "input should be 3 dim"

    D, H, W = image.shape
    y_ = image.clone()
    y_[:gap] = 0
    y_[:, :gap] = 0
    y_[:, :, :gap] = 0
    y_[D - gap:] = 0
    y_[:, H - gap] = 0
    y_[:, :, W - gap] = 0

    return y_

class Contour: 
    def __init__(self, points, z, origin, dir_mat, gtvolume_resolution, N, cls):
        self.points = points 
        self.z = z
        self.origin = origin
        self.dir_mat = dir_mat
        self.gtvolume_resolution = gtvolume_resolution
        self.N = N
        self.cls = cls
        
        self.center = self.get_center(points)

    def get_center(self, cont1): 
        A = cont1[None]
        B = cont1[:, None] 
        dist = np.sum((A-B)**2, axis=2)
        opp_idx = np.argsort(dist, axis=1)
        opp_idx = opp_idx[:, -1]
        cont1 = 0.5 * (cont1 + cont1[opp_idx])
        center1 = np.mean(cont1, axis=0)
        return center1

class Slice:
    def __init__(self, image, z, top_left, scale_factor, image_original, dicom_resolution):
        self.image = image 
        self.image_original = image_original
        self.z = z
        self.top_left = top_left
        self.top_left2d = None
        self.image_color = None 

        self.masks = {}
        self.contours = [] 
        self.scale_factor = scale_factor
        self.dicom_resolution = dicom_resolution
         
    def get_z(self):
        return self.z

    def compute_contour_origin(self):  
        
        origins = [] 
        for contour in self.contours:
            origins += [contour.origin[None]]

        if len(self.contours) > 0:
            origins = np.concatenate(origins, axis=0)
            origins = np.unique(origins, axis=0)
            if origins.shape[0] > 1:
                self.origin = None 
                return 0
            # assert origins.shape[0] == 1, "incosistant resolution"
            if origins.shape[0] > 1: 
                print('no unique origin')
            self.origin = origins.squeeze() 
        else:
            print(f'no origin in contours {self.z}') 
            self.origin = None
        return 1

class Stack:
    def __init__(self, slices, name, type, fill=False):
        self.slices = slices
        self.name = name
        self.type = type
  
        self.min_max()
 
 

    def detach(self):
        for cls, contours in self.all_contours.items():
            for slice_ids, contour in contours.items():   
                self.all_contours[cls][slice_ids] = contour.detach()

    def compute_volumes(self, filled=False):
         
         
        image_vol = self._to_stack() 
         
        image = torch.from_numpy(image_vol).float().cuda()

        print(self.slices[0].dicom_resolution)
        h_resolution, w_resolution, d_resolution = self.slices[0].dicom_resolution
        # embed()
        if self.name == 'Study-2':
            d_resolution= 2  * 0.75
            h_resolution = h_resolution * 0.75
            w_resolution = w_resolution * 0.75
        elif self.name == 'Study-1':
            d_resolution= 3 * 0.75
            h_resolution = h_resolution * 0.75
            w_resolution = w_resolution * 0.75
        D, H, W = image.shape  
        D = int(D * d_resolution) # 
        H = int(H * h_resolution) # 
        W = int(W * w_resolution)  #   
        # we resample such that 1 pixel is 1 mm in x,y and z directiions
        base_grid = torch.zeros((1, D, H, W, 3))
        w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
        h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
        d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)
        base_grid[:, :, :, :, 0] = w_points
        base_grid[:, :, :, :, 1] = h_points
        base_grid[:, :, :, :, 2] = d_points
        
        grid = base_grid.cuda()
            
        image = F.grid_sample(image[None, None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0, 0]
               
        image_vol = image.cpu().numpy()
        image_vol = np.swapaxes(image_vol, 0, 2)



        # embed()
        # a = image_vol - image_vol.min()
        # a = img - img.min()
        # a = a/a.max()  
        # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/tw1.tif', np.uint8(255*a))

        if self.name == 'Study-2':
            # 1 resolution
            # image_vol = image_vol[225:280]
            # image_vol = image_vol[::4, 125-64:125+64, 64-64:64+64]

            # 0.75 resolution
            image_vol = image_vol[170:210]
            image_vol = image_vol[::3, 96-64:96+64, 64-64:64+64] 
            d_resolution = d_resolution * 3
            # no resampling
            # image_vol = image_vol[165:210]
            # image_vol = image_vol[::3, 96-64:96+64, 64-64:64+64]
        elif self.name == 'Study-1':
            # 1 resolution
            # image_vol = image_vol[255:317]
            # image_vol = image_vol[::5, 185-64:185+64, 85-64:85+64]

            # 0.75 resolution
            image_vol = image_vol[190:230]
            image_vol = image_vol[::3, 140-64:140+64, 64-64:64+64]
            d_resolution = d_resolution * 3
            # image_vol = image_vol[:, :, ::-1] 

            # image_vol = image_vol[190:230]
            # image_vol = image_vol[:, 370-64:370+64, :]
        else:
            print('no supported yet')
  
        for p in range(len(self.slices)):
            self.slices[p].dicom_resolution[0] = h_resolution
            self.slices[p].dicom_resolution[1] = w_resolution
            self.slices[p].dicom_resolution[2] = d_resolution

        self.image_vol = image_vol


        # if filled: # TODO: useless if? check 
        #     mask_wall = np.uint8(np.zeros_like(image_vol))
        #     mask_scar = np.uint8(np.zeros_like(image_vol))
        #     mask_wall[masks_vol[2]] = 2  
        #     mask_wall[masks_vol[1]] = 1   
        #     if 9 in masks_vol.keys():
        #         mask_scar[masks_vol[9]] = 1 
        #     if 10 in masks_vol.keys():
        #         mask_scar[masks_vol[10]] = 1
               
        #     # mask_scar[np.logical_or(masks_vol[9],masks_vol[10])] = 1  
        # else: 
        #     mask_wall = np.uint8(np.zeros_like(image_vol))
        #     mask_scar = np.uint8(np.zeros_like(image_vol))
        #     mask_wall[masks_vol[2]] = 2  
        #     mask_wall[masks_vol[1]] = 1    
        #     if 9 in masks_vol.keys():
        #         mask_scar[masks_vol[9]] = 1 
        #     if 10 in masks_vol.keys():
        #         mask_scar[masks_vol[10]] = 1
        #     # mask_scar[np.logical_or(masks_vol[9],masks_vol[10])] = 1  
         
        # if np.any(mask_wall.sum(1).sum(1)==0):
            
        #     pattern = np.float32(mask_wall.sum(1).sum(1)==0)
        #     diff =  pattern[1:] -  pattern[:-1]
        #     # diff = diff[1:-1]
        #     if np.any(diff>0):
        #         print('missing contours in some slices!') 
        #         print(pattern)
        # self.image_vol = image_vol
        # self.mask_scar = mask_scar
        # self.mask_wall = mask_wall 

         
        # _, H, W = mask_wall.shape
        # y = np.arange(H)
        # x = np.arange(W)
        # X, Y = np.meshgrid(x, y)
        # X = np.ravel(X)
        # Y = np.ravel(Y)
        # grid = np.concatenate([X[:, None], Y[:, None]], axis=1)
        # max_ = np.sqrt(H*H + W*W)
        # mask_center = np.float32(np.zeros_like(mask_wall))
        
        # for slice_id, center in self.all_centers[1].items():
        #     center = center.numpy() 
        #     resolution = self.slices[slice_id].dicom_resolution[:2][None] 
        #     distmap = 1-np.sqrt(np.sum((grid - resolution * center) ** 2, axis=1))/max_
        #     mask_center[slice_id, Y, X] = distmap

        # self.mask_center = mask_center
         
        # from skimage import io 
        # mask_center = np.float32(mask_center)
        # a = (mask_center - mask_center.min())/(mask_center.max()-mask_center.min())

        # a = (image_vol - image_vol.min())/(image_vol.max()-image_vol.min())
        # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/pred_voxels.tif', np.uint8(255*a))
        # io.imsave('/cvlabdata2/cvlab/datasets_udaranga/outputs/pred_voxels2.tif', np.uint8(67*mask_wall))
        # for w in mask_wall:
        #     x,y = np.where(w>0)
        #     print(f'{np.mean(y)}, {np.mean(x)}')
        # embed()

    def get_gttool_resolution(self):
        res = []
        for slice in self.slices:
            for contour in slice.contours:
                res += [contour.gtvolume_resolution[None]]
        res = np.concatenate(res, axis=0)
        res = np.unique(res, axis=0)
        # assert res.shape[0] == 1, "incosistant resolution"
        return res

    def get_dicom_resolution(self):
        res = []
        for slice in self.slices:
            res += [slice.dicom_resolution[None]] 
        res = np.concatenate(res, axis=0)
        res = np.unique(res, axis=0)
        # assert res.shape[0] == 1, "incosistant resolution"
        return res
    

    def get_dir_mat(self): 
        dir_mat = []
        for slice in self.slices:
            for contour in slice.contours:
                dir_mat += [contour.dir_mat[None]] 
        dir_mat = np.concatenate(dir_mat, axis=0)
        dir_mat = np.unique(dir_mat, axis=0)
        assert dir_mat.shape[0] == 1, "incosistant direction matrix"
        return dir_mat.squeeze()

    def _get_all_points(self): 
        # self.points3d = {}
        for slice_id, slice in enumerate(self.slices):
            for contour in slice.contours: 
                # Contour points from Gt Volumes
                points2d = torch.from_numpy(contour.points)
                center2d = torch.from_numpy(contour.center)[None]

                # Direction matrix
                dir_mat = torch.from_numpy(contour.dir_mat) 

                # points in 3D - not used at the moment
                # px = points2d[:, 0][:, None] * dir_mat[:,0][None]
                # py = points2d[:, 1][:, None] * dir_mat[:,1][None]
                # contour_3d = px + py + contour.origin

                # points coordinates w.r.t slice/image coordinates (2d)
                topleft_x = torch.sum(dir_mat[:,0] * slice.top_left)   
                topleft_y = torch.sum(dir_mat[:,1] * slice.top_left) 
                self.top_left2d = torch.tensor([topleft_x, topleft_y])[None]

                points2d = (points2d - self.top_left2d)/contour.gtvolume_resolution[:2][None] 
                contour_2d = points2d * torch.tensor([1, -1])[None] # flip y coordinate
 
                center2d = (center2d - self.top_left2d)/contour.gtvolume_resolution[:2][None] 
                center_2d = center2d * torch.tensor([1, -1])[None] # flip y coordinate
                
                if contour.cls in self.all_contours: 
                    self.all_contours[contour.cls][slice_id] = contour_2d
                    self.all_centers[contour.cls][slice_id] = center_2d 
                else:
                    self.all_contours[contour.cls] = {} 
                    self.all_contours[contour.cls][slice_id] = contour_2d  

                    self.all_centers[contour.cls] = {} 
                    self.all_centers[contour.cls][slice_id] = center_2d  
  

         
        
    def _draw_contours(self, fill=True):  

        for slice in self.slices: 
            slice.masks = {}

        for cls, contours in self.all_contours.items():
            for slice_ids, contour in contours.items(): 
                if cls not in self.slices[slice_ids].masks.keys():
                    self.slices[slice_ids].masks[cls] = np.zeros_like(self.slices[slice_ids].image)

                resolution = torch.from_numpy(self.slices[slice_ids].dicom_resolution[:2][None]) 
                contour = contour * resolution
                self.all_contours[cls][slice_ids] = contour 

                if fill:
                    self.slices[slice_ids].masks[cls] = cv2.fillPoly(self.slices[slice_ids].masks[cls] , [np.int64(contour.detach().cpu().numpy())], (255, 0, 0))
                else:
                    self.slices[slice_ids].masks[cls] = cv2.drawContours(self.slices[slice_ids].masks[cls], [np.int64(contour.detach().cpu().numpy())], -1, (255, 0, 0), 1)

    def min_max(self):
 
        images = []
        for slice in self.slices:
            images += [slice.image[None]]
        
        images = np.concatenate(images, axis=0) 
        self.minmax = [images.min(), images.max()]
        # self.image_color = np.uint8(255*(image_color - image_color.min())/(image_color.max()-image_color.min())) 
    
    def _to_stack(self):    
         
        images = []
        sizes = []
        for slice in self.slices: 
            images += [slice.image[None]] 
            sizes += [np.array(slice.image.shape)[None]]
        sizes = np.unique(np.concatenate(sizes, axis=0),axis=0)
        H, W = sizes[0]
        if len(sizes) > 1: 
            for i, img in enumerate(images):
                _, h, w = img.shape
                images[i] = crop(img[0], (H, W), (h//2, w//2))[None]
        images = np.concatenate(images, axis=0) 
 
        return images

def xml2contours(path, slices):
    # embed()
    # path = '/cvlabsrc1/cvlab/datasets_udaranga/datasets/3d/CHUV/dataset/1-FB-ICM-Learn/FB_CMR examination_032/old'

    files = listdir(path) 


    xml_file = [f for f in files if 'xml' in f]

    if len(xml_file) > 1 or len(xml_file)==0:
        return False

    # embed()
    tree = ET.parse(f'{path}/{xml_file[0]}')
    root = tree.getroot()

    W, H, D = stringtoarray(root.find('Dataset').findtext('Dimensions'))[:3]
 
    classes = []
    for elem in root:
        if elem.findtext('IndexType') is not None:
            classes += [int(elem.findtext('IndexType'))]
    classes = np.unique(np.array(classes))
    # classes = np.array([1])
 
    all_contours = {}  
    printed = True
    for cls in classes:
        contours = [] 
        for elem in root:  
            if elem.findtext('Vertices') is not None and int(elem.findtext('IndexType'))==cls: 

                res_x = float(elem.find('HostStack').find('gtBdImageStack').findtext('VoxelSizeX'))
                res_y = float(elem.find('HostStack').find('gtBdImageStack').findtext('VoxelSizeY'))
                res_z = float(elem.find('HostStack').find('gtBdImageStack').findtext('VoxelSizeZ'))
                
                resolution = np.array([res_x/slices[0].scale_factor, res_y/slices[0].scale_factor, res_z])  

                origin = stringtoarray(elem.find('gtBdDataObject').findtext('CoordinatesOrigin'))
                RL = stringtoarray(elem.find('gtBdDataObject').findtext('DirectionCosinesRL'))[None]
                AP = stringtoarray(elem.find('gtBdDataObject').findtext('DirectionCosinesAP'))[None]
                FH = stringtoarray(elem.find('gtBdDataObject').findtext('DirectionCosinesFH'))[None]

                dir_mat = np.concatenate([RL, AP, FH], axis=0) 
                # contour = elem.findtext('Vertices').split() 
                # contour = np.array([float(c) for c in contour]).reshape(-1, 2)
                points = stringtoarray(elem.findtext('Vertices')).reshape(-1, 2) 
                N = int(elem.findtext('NumberOfVertices'))  
        
                n = dir_mat[:,2] 
                z = np.sum(n * origin)   

                index = elem.findtext('IndexSlice')
                
                # embed()
                # print(f'{z} {index} {-42.004685794900006-z}')
                
                contour = Contour(points, z, origin, dir_mat, resolution, N, cls)
                # 
                found = False 
                for slice in slices: 
                    # print(f'{slice.z}') 
 
                    slice_z = slice.z

                    if slice_z * z > 0:
                        dist = np.abs(slice_z - z) 
                    else:
                        dist = np.abs(slice_z + z) 
                    if dist < 0.01:  
                        slice.contours += [contour] 
                        found = True
                
                # embed()
                # print(found)
                if not found: 
                    slices[int(index)].contours += [contour] 
    return True

def resacle(x, scale_factor, mode='nearest'): 
    is_numpy = False
    if scale_factor == 1:
        # print('o')
        return x
    shape = torch.tensor(x.shape) 
    shape = torch.cat([torch.tensor([1, 1]), shape])
    shape[3:] = scale_factor * shape[3:]
    grid = F.affine_grid(torch.eye(4)[:3][None].cuda(), tuple(shape), align_corners=True)
 
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).cuda().float()
        is_numpy = True

    x = F.grid_sample(x[None, None], grid, mode=mode, padding_mode='zeros', align_corners=True)[0, 0]
    x = x.cpu().numpy() if is_numpy else x
    return x

def load_snapshots(path, scale_factor=1):
    slices = []
    fs = FileSet(f'{path}/DICOMDIR')    
    count = 0
    for i, instance in enumerate(fs):
        # Load the corresponding SOP Instance dataset
        ds = instance.load() 
        try: 
            if int(ds.SeriesNumber) == 65001:
                print('-------')
                count += 1
                print(count)
            # print(ds.SeriesNumber)
        except AttributeError: 
            print('missing')       
            
def load_stack(path, scale_factor=1):  
    
    slices = [] 
    fs = FileSet(f'{path}/DICOMDIR')     
    # ds.trait_names() -> list all traits (tags) 
    resolutions = [] 
    for i, instance in enumerate(fs):
        # Load the corresponding SOP Instance dataset
        ds = instance.load()   
        
        # if ds.SequenceName == '*fl3d1_22': 
            
        image_original = np.float32(ds.pixel_array) 
        image = resacle(image_original[None], scale_factor=scale_factor, mode='nearest').squeeze()  
        z = np.array(ds.SliceLocation)
        top_left = np.array(ds.ImagePositionPatient) 
        try:
            dicom_resolution = np.array([ds.PixelSpacing[0]/scale_factor, ds.PixelSpacing[1]/scale_factor, ds.SpacingBetweenSlices])
        except AttributeError: 
            # !!!not the correct way to get z res!!!
            dicom_resolution = np.array([ds.PixelSpacing[0]/scale_factor, ds.PixelSpacing[1]/scale_factor, ds.SliceThickness])
        # resolutions += [dicom_resolution[None]]
        slice = Slice(image, z, top_left, scale_factor, image_original, dicom_resolution)

        slices += [slice]   
    # if len(resolutions) > 0:
    #     resolutions = np.concatenate(resolutions, axis=0)
    #     unique_res = np.unique(resolutions, axis=0)
    #     if unique_res.shape[0] > 1:
    #         print('Warning!')
    #         print(unique_res) 
            
    
    slices = sorted(slices, key=lambda x:x.z)

    for i, slice in enumerate(slices):
        slice.z_index = i
     
    return slices 
 



def get_scar_regions_volume(image_vol, mask_scar, mask_wall, th1, th2):
    image = np.float32(image_vol)
    image_msk = image[mask_scar]
    mean_ = np.mean(image_msk)
    std_ = np.std(image_msk)
    scar_th1 = mask_wall * (image > (mean_ + th1*std_))
    scar_th2 = mask_wall * (image > (mean_ + th2*std_))
    return scar_th1, scar_th2

def get_scar_regions_slice(image_vol, mask_scar, mask_wall, th1, th2):
    image = np.float32(image_vol)
    scars_th1 =[]
    scars_th2 =[]
    for (img, msk_scar, msk_wall) in zip(image, mask_scar, mask_wall):
        image_msk = img[msk_scar]
        mean_ = np.mean(image_msk)
        std_ = np.std(image_msk)
        scars_th1 += [(msk_wall * (img > (mean_ + th1*std_)))[None]]
        scars_th2 += [(msk_wall * (img > (mean_ + th2*std_)))[None]]

    scar_th1 = np.concatenate(scars_th1,axis=0)
    scar_th2 = np.concatenate(scars_th2,axis=0)
    return scar_th1, scar_th2