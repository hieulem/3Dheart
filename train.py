import logging
import torch
# from torch.utils.tensorboard import SummaryWriter
from utils.utils_common import Modes
 
import numpy as np
import time 
import wandb
from IPython import embed
import time 
from collect_env import run
logger = logging.getLogger(__name__)


 
class Trainer(object):

 
    def training_step(self, data, iteration):
        # Get the minibatch 
         
        self.optimizer.zero_grad() 
        pred, data = self.net(data)  
        loss, log = self.net.loss(pred, data)  
        loss.backward()
        self.optimizer.step()  
        # embed()

        return log

    def __init__(self, net, trainloader, optimizer, numb_of_itrs, eval_every, save_path, evaluator):

        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer 

        self.numb_of_itrs = numb_of_itrs
        self.eval_every = eval_every
        self.save_path = save_path 

        self.evaluator = evaluator 




    def train(self, start_iteration=0, use_wandb=False):

        print("Start training...") 
 
        self.net = self.net.train()
        iteration = start_iteration 

        print_every = 1
        continue_training = True
        while continue_training:  # loop over the dataset multiple times
            for data in self.trainloader: 
                # a = run('nvidia-smi')
                # mem = int(a[1].split('\n')[9][35:40])
                # print(f'{iteration}:  {mem}')

                if iteration % self.eval_every == 0:  # print every K iterations
                    self.evaluator.evaluate(iteration) 
                
                # training step 
                loss = self.training_step(data, iteration)

                if iteration % print_every == 0:
                    log_vals = {}
                    for key, value in loss.items():
                        log_vals[key] = value / print_every 
                    log_vals['iteration'] = iteration
                    if use_wandb:
                        wandb.log(log_vals) 

                iteration = iteration + 1 
                if iteration == self.numb_of_itrs: 
                    continue_training = False
                    break   
 
        logger.info("... end training!")
 