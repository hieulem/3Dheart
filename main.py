 
import os
GPU_index = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index

  
import logging
import torch
import numpy as np
from train import Trainer
from evaluate import Evaluator  

from shutil import copytree, ignore_patterns
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils_common import Modes 
import wandb
from IPython import embed 
from utils.utils_common import mkdir
 
from configs.config_CHUV import load_config
# from configs.config_TW import load_config
# from model.hearts import HEARTS as network
from model.hearts2 import HEARTS as network


# from configs.config_full_heart import load_config
# from model.voxel2mesh_pp import Voxel2Mesh as network

# from model.regvoxel2mesh_pp import RegVoxel2Mesh as network 
# from model.localizernet import LocalizerNet as network
# from model.unet import UNet as network



logger = logging.getLogger(__name__)

 
def init(cfg):

    save_path = cfg.save_path + cfg.save_dir_prefix + str(cfg.experiment_idx).zfill(3)
    
    mkdir(save_path) 
 
    trial_id = (len([dir for dir in os.listdir(save_path) if 'trial' in dir]) + 1) if cfg.trial_id is None else cfg.trial_id
    trial_save_path = save_path + '/trial_' + str(trial_id) 

    if not os.path.isdir(trial_save_path):
        mkdir(trial_save_path) 
        copytree(os.getcwd(), trial_save_path + '/source_code', ignore=ignore_patterns('*.git','*.txt','*.tif', '*.pkl', '*.off', '*.so', '*.json','*.jsonl','*.log','*.patch','*.yaml','wandb','run-*'))

  
    seed = trial_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True  # speeds up the computation 

    return trial_save_path, trial_id

def main():
  
    exp_id = 253

    # 5 unet
    # 6 unet with contrast, brightness and shift augmentation
    # 7 unet 5 steps
    # 8 high res added but not used
    # 9 high res added for output
    # 10 high res output as well
    # 11 two classes
    # 12 high res image as well
    # 13 128 x 128 x 128
    # 14 64 x 64 x 64
    # 15 liver seg
    # 16 Liver two class (liver)
    # 17, 18 liver tumor 
    # 19 liver
    # 17-9 tumor
    # 17-10 liver
    # 20

    # 30: normal image
    # 31: hist normalized
    # 32: manual clipping
    # 33: hist normalized, chamf dist
    # 34: low regularization
    # 35: high regularization again, 31
    # 36: unet
    # 40: dataset 35 
    # 41: dataset 45
    # 42: dataset 56

    # 50: contour coordinates useds
    # 51: order of class changed
    # 52: outer points
    # 53: inner points

    # 54: standard sampler
    # 55: mc sampler res 1
    # 56: mc sampler res 2
    # 58: FixedNeighbourhoodSampling, res 2
    # 59: FixedNeighbourhoodSampling, res 2, 3 extra coord channels

    # 60: 59 again,
    # 61: 60 with 35 training samples
    # 62: 60 with dataset v4
    # 63: after registration 
    # 64: control, 60
    # 65: clip zero y planes from x
    # 66: registered dataset (v5), no slice shifts augmentations, - much better, but some anomolous registration present
    # 67: non-registered dataset (v6), no slice shifts augmentations, 
    # 68: non-registered dataset (v6), with slice shifts augmentations - 10pixels, 
    # 69: non-registered dataset (v6), with slice shifts augmentations - 10pixels, no smooth regularization
    # 70: non-registered dataset (v6), with slice shifts augmentations - 2pixels, 
    # 71: non-registered dataset (v6), with slice shifts augmentations - 0pixels, 

    # 72: resolution * vertices
    # 73: resolution * (shape-1) * vertices
    # 74: 73, fixed
    # 75: 74 baselines, no normalization
    # 76: 75 again, basline check

    # 80: normalized vertices, standard loss
    # 81: 100, 100, for chamfer and ce loss
    # 82: 100 for ce loss only
    # 83: slice shift augme ntation - 5pixels
    # 84: slice shift augme ntation - 0pixels
    # 85: slice shift augme ntation - 1pixels
    # 86: slice shift augme ntation - 2pixels

    # 90: regvoxel2mesh
    # 91: load pretrained voxel2mesh and freeze weights
    # 92: load pretrained voxel2mesh and weights are not freezed
    # 93: baseline 80
    # 94: frozen voxel2mesh
    # 95: frozen voxel2mesh + with 1 step on regnet
    # 96: frozen voxel2mesh + shifted x + with 1 step on regnet - wrong exp
    # 97: frozen voxel2mesh + shifted x + with 3 step on regnet - wrong exdp
    # 98: unfrozen voxel2mesh + shifted x - wrong exp
    # 99: unfrozen voxel2mesh + shifted x + 3 steps
    # 100: frozen voxel2mesh + shifted x + 3 steps
    # 101: voxel2mesh continue training
    
    # 102: frozen voxel2mesh + no-shifted x + 3 steps
    # 103: frozen voxel2mesh + shifted x + 3 steps --- good single sample
    # 104: unfrozen voxel2mesh + shifted x + 3 steps --- good single sample
    # 105: frozen voxel2mesh + shifted x + 3 steps + full dataset
    # 106: unfrozen voxel2mesh + shifted x + 3 steps + full dataset
    # 107: just voxel2mesh
    # 105-2: frozen voxel2mesh + shifted x + 3 steps + one sample dataset
    # 108: unfrozen voxel2mesh + shifted x + 3 steps + one sample dataset
    # 109: 105-2 but with unlabeled slices clipped + one sample dataset
    # 110: 109 with unfrozen unet + one sample dataset
    # 111: 105-2 but with unlabeled slices clipped + full dataset
    # 112: 109 with unfrozen unet + full dataset
    # 113: just v2m
    # 114: 112, unfrozen bug fixed

    # 152: 4 steps, 8 first layer, 64 channels from bottom, 128 cropped
    # 153: 5 steps, 16 first layer, 128 channels from bottom, 128 cropped
    # 154: 6 steps, 16 first layer, 128 channels from bottom, 320x256 cropped
    # 155: last trial, localizer, slice-shift and v2m++ working together
    # 160: frozen locallizer, with slice shift + unfrozen v2m
    # 161: frozen locallizer, with slice shift + unfrozen v2m

    # 170: scarnet, 3 classes
    # 171: scarnet, 3 classes, no weights
    # 172: weighted, 3 classes
    # 173: weighted, 3 classes, wall as input to scarnet
    # 174: scarnet, 3 classes, no weights, wall as input to scarnet
    # 175: mse, masked
    # 176: mse, no mask


    # 190: first two pretrained
    # 191: from scratch full network
    # 192: repeat 190
    # 193: repeat 191
    # 195: CHUV3 first trial
    # 196: localizer gt fixed
    # 197: scarnet inference time augmentation added

    # 200: 197, repeat
    # 201: 197 without slice registration -- bad
    # 202: 197 without test time augmentation -- not that bad

    # 203: new brightness/contrast augmentation
    # 204: 203 with no heartwall input to scarnet --
    # 205: binary cross entropy

    # 206: repeat 200 16K
    # 207: v2m, 2 graph conv layers mem 10K - this is enough, no change in performance *****
    # 208: 4 steps in scarnet  15K -  performance doesn't degrade, but no significant improvement in memeory use
    # 209: first layer count 8 in scarnet 14K - bad idea, degrades the performance
    # 210: 96 patch size

    # 211: scarnet with regression
    # 213: scarent with regression with masked mse

    # 220: 200 repeat with graphconv 2 layers
    # 221: 220 without v2m input to scarnet

    # 232: 1st exp with localizer, trained with gt centers
    # 233: use pred centers during eval

    # 240: 220 repeat with localizer
    # 241: 240 with image multiplied by LV 
    # 242: 240 repeat with class id fixed

    # 250: heart2
    # 251: clipping unannotated slices back to before flip line
    # 252: training shift augmentations added to heart2.py forward
    # 253: no lv input to scarnet

    # 300: full heart
    # 301: full heart with 60 sample dataset

    # TOCHECK
    # LearntNB sampling
    # Loss regularizer effect

    
 
    # Initialize
    cfg = load_config(exp_id)
    trial_path, trial_id = init(cfg) 
 
    print('Experiment ID: {}, Trial ID: {}'.format(cfg.experiment_idx, trial_id))

    print("Create network")
    classifier = network(cfg)
    classifier.cuda()
  
    use_wandb = False
    # if exp_id > 10 and cfg.trial_id is None:
    wandb.init(name='Experiment_{}/trial_{}'.format(cfg.experiment_idx, trial_id), project="voxel2mesh-pp", dir=trial_path)
    use_wandb = True
 
    print("Initialize optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=cfg.learning_rate)  
  
    print("Load pre-processed data") 
    data_obj = cfg.data_obj 
    data = data_obj.quick_load_data(cfg, trial_id)
 
    # data_ = []  
    # for d in data[Modes.TESTING].samples: 
    #     if '035' in d.name: # 080, 009 
    #         data_ += [d]  
    # data[Modes.TRAINING].samples = data_
    # data[Modes.TESTING].samples = data_ 

    # data[DataModes.TRAINING].samples = data[DataModes.TRAINING].samples[:35]
    loader = DataLoader(data[Modes.TRAINING], batch_size=classifier.config.batch_size, shuffle=True)
  
    print("Trainset length: {}".format(loader.__len__()))

    print("Initialize evaluator")
    evaluator = Evaluator(classifier, optimizer, data, trial_path, cfg, data_obj) 

    print("Initialize trainer")
    trainer = Trainer(classifier, loader, optimizer, cfg.numb_of_itrs, cfg.eval_every, trial_path, evaluator)

    if cfg.trial_id is not None:
        # To evaluate a pretrained model, uncomment line below and comment the line above 
        print("Loading pretrained network")
        save_path = trial_path + '/best_performance3/model.pth'
        checkpoint = torch.load(save_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        evaluator.evaluate(epoch)
    else: 
        trainer.train(start_iteration=0, use_wandb=use_wandb) 



if __name__ == "__main__": 
    main()