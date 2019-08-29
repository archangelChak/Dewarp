from torch import nn

from UNet.UNet import Encoder, Decoder
from utils.layers import conv3x3

class DocUNet(nn.Module):
    """
    Creates 2 stacked U-net graph where output of the first one is used as input of the second one. First U-net solves the problem of
    binarization, and the second solves the problem of detection.

    :param num_classes: int: number of classes for uotput tensor
    :param num_blocks: number of blocks in encoder and decoder of the Unet implementation,
            be careful that shape of the input tensor divisible by 2**num_blocks
    :param num_filters:  number of filters in the first layer in encoder
    :param in_channels: int: number of channels in input tensor (image)
    :return: (tf.Tensor, tf.Tensor): output of the last layer of each U-net
    """
    def __init__(self, num_classes=1, in_channels=1, num_filters=4, num_blocks=5):
        super().__init__()

        self.encoder = Encoder(in_channels, num_filters, num_blocks)
        self.decoder = Decoder(num_filters, num_blocks - 1)
        self.concat =  torch.cat
        self.encoder1 = Encoder(num_filters+num_classes, num_filters, num_blocks)
        self.decoder1 = Decoder(num_filters, num_blocks - 1)
        self.final = nn.Conv2d(num_filters, num_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.concat((x , self.final(x)),1)
        x = self.encoder1(x)
        x = self.decoder1(x)
        x = self.final(x)
        return x