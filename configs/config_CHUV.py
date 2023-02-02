import numpy as np
import torch 

from data.chaos import Chaos
from data.hippocampus import Hippocampus 
from data.CHUV.heart import Heart

class Config():
    def __init__(self):
        super(Config, self).__init__()


def load_config(exp_id=None):
      
    cfg = Config()
    ''' Experiment '''
    cfg.experiment_idx = exp_id 
    cfg.trial_id = None
    cfg.device = "cuda"
    
    cfg.save_dir_prefix = 'Experiment_' # prefix for experiment folder
    cfg.name = 'voxel2mesh'
    # cfg.name = 'unet'
    # cfg.name = 'localizer'

    ''' 
    **************************************** Paths ****************************************
    save_path: results will be saved at this location
    dataset_path: dataset must be stored here.
    ''' 
    cfg.dataset_path = '/cvlabsrc1/cvlab/datasets_udaranga/datasets/3d/CHUV/dataset'
    # cfg.dataset_path = '/cvlabsrc1/cvlab/datasets_udaranga/datasets/3d/CHUV/dataset_snapshot'
    # cfg.dataset_path = '/cvlabsrc1/cvlab/datasets_udaranga/datasets/3d/Task04_Hippocampus'
    cfg.data_obj = Heart()
    # cfg.data_obj = Hippocampus()

    cfg.save_path = '/cvlabdata2/cvlab/datasets_udaranga/experiments/voxel2mesh_pp/'



    assert cfg.save_path != None, "Set cfg.save_path in config.py"
    assert cfg.dataset_path != None, "Set cfg.dataset_path in config.py"
    assert cfg.data_obj != None, "Set cfg.data_obj in config.py"

    ''' 
    ************************************************************************************************
    ''' 


    ''' Dataset '''  
    # input should be cubic. Otherwise, input should be padded accordingly.
    cfg.patch_shape = (32, 128, 128) 
    cfg.patch_shape_before_crop = (32, 320, 256) 
    cfg.ndims = 3 
    cfg.brightness_factor = 40
    cfg.contrast_factor = 0.2
    cfg.shift_factor = 15   
    cfg.scale_factor = 5
    cfg.regress_scar = False

    ''' Model '''
    cfg.register_slices = True
    cfg.first_layer_channels = 16
    cfg.num_input_channels = 4
    cfg.steps = 4

    # Only supports batch size 1 at the moment. 
    cfg.batch_size = 1 

 
    cfg.class_ids = [1,2] # [1,2]
    cfg.num_classes = len(cfg.class_ids) + 1
    cfg.lv_num_classes = 3
    cfg.scar_num_classes = 5
    cfg.batch_norm = True  
    cfg.graph_conv_layer_count = 2

  
    ''' Optimizer '''
    cfg.learning_rate = 1e-4

    ''' Training '''
    cfg.numb_of_itrs = 200000
    cfg.eval_every = 1000 # saves results to disk
    cfg.eval_metric = 'jaccard_scar'

    # ''' Rreporting '''
    # cfg.wab = True # use weight and biases for reporting
    
    return cfg