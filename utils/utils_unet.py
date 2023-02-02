import torch.nn as nn  
from IPython import embed 

class UNetLayer(nn.Module):
    """ U-Net Layer """
    def __init__(self, num_channels_in, num_channels_out, ndims, batch_norm=False, kernel_size=3, padding=1):

        super(UNetLayer, self).__init__()

        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        batch_nrom_op = nn.BatchNorm2d if ndims == 2 else nn.BatchNorm3d

        conv1 = conv_op(num_channels_in,  num_channels_out, kernel_size=kernel_size, padding=padding)
        conv2 = conv_op(num_channels_out, num_channels_out, kernel_size=kernel_size, padding=padding)

        bn1 = batch_nrom_op(num_channels_out)
        bn2 = batch_nrom_op(num_channels_out)
        self.unet_layer = nn.Sequential(conv1, bn1, nn.ReLU(), conv2, bn2, nn.ReLU())

    def forward(self, x):
        return self.unet_layer(x)