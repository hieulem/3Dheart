from itertools import chain
import torch.nn as nn
from utils.utils_common import crop_and_merge
import torch
from IPython import embed
from utils.utils_unet import UNetLayer 
 

class ScarNet2D(nn.Module):
    """ The U-Net. """
    def __init__(self, config):

        super(ScarNet2D, self).__init__()
        self.config = config

        # steps = config.steps 
        steps = 5
        first_layer_channels = 16 # 8
        num_input_channels = 2
              
        assert config.ndims ==3, Exception("Invalid nidm: {}".format(config.ndims))
        self.max_pool = nn.MaxPool3d([1,2,2])  
        ConvLayer = nn.Conv3d 
        ConvTransposeLayer = nn.ConvTranspose3d 
   
        '''  Down layers '''
        down_layers = [UNetLayer(num_input_channels, first_layer_channels, config.ndims, kernel_size=[1,3,3], padding=[0,1,1])]
        for i in range(1, steps + 1):
            lyr = UNetLayer(first_layer_channels * 2**(i - 1), first_layer_channels * 2**i, config.ndims, kernel_size=[1,3,3], padding=[0,1,1])
            down_layers.append(lyr)

        ''' Up layers '''
        up_layers = [] 
        for i in range(steps - 1, -1, -1):  
            upconv = ConvTransposeLayer(in_channels=first_layer_channels * 2**(i+1), out_channels=first_layer_channels * 2**i, kernel_size=[1,2,2], stride=[1,2,2])
            lyr = UNetLayer(first_layer_channels * 2**(i + 1), first_layer_channels * 2**i, config.ndims, kernel_size=[1,3,3], padding=[0,1,1])
            up_layers.append((upconv, lyr))

        ''' Final layer '''
        # if self.config.regress_scar:
        #     final_layer = ConvLayer(in_channels=first_layer_channels, out_channels=1, kernel_size=1)
        # else:
        final_layer = ConvLayer(in_channels=first_layer_channels, out_channels=config.scar_num_classes, kernel_size=1)
 

        self.down_layers = down_layers
        self.up_layers = up_layers

        self.down = nn.Sequential(*down_layers)
        self.up = nn.Sequential(*chain(*up_layers)) 
        self.final_layer = final_layer

        print(f'regress scars: {self.config.regress_scar}')
        
    def forward(self, x):   
        # first layer
        x = self.down_layers[0](x)
        down_outputs = [x]

        # down layers
        for unet_layer in self.down_layers[1:]: 
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x) 
 
        # up layers
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers, down_outputs[-2::-1]):
            x = upconv_layer(x)
            x = crop_and_merge(down_output, x)
            x = unet_layer(x)

        pred = self.final_layer(x)

        return pred


    def loss(self, pred, data):

        # pred = self.forward(data)
 
        
        # weights = torch.ones(3).cuda()
        # for t in target.unique(): 
        #     weights[t] = torch.sum(target==t) 
        # weights = weights.sum()/(len(target.unique())*weights)
        # weights = weights/weights.sum()   
        # CE_Loss = nn.CrossEntropyLoss(weight=weights)
        if self.config.regress_scar:  
            target = data['y_scar'] 
            mask = data['y_myocardium'].long() 
            MSE_Loss = nn.MSELoss(reduction='none')  
            loss = MSE_Loss(pred[:, 0], target) # target.squeeze()
            loss = torch.mean(loss * mask)
            log = {"loss_scar_mse": loss.detach()}
        else:
            target = data['y_scar'].long()  
            CE_Loss = nn.CrossEntropyLoss()
            loss = CE_Loss(pred,  target)
            log = {"loss_scar_ce": loss.detach()}

        return loss, log