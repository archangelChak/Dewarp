import torch

import torch.nn as nn
from utils.layers import conv3x3

class EncoderBlock(nn.Module):
    """
    Creates encoder block U-net based architecture

    :param in_channels: number of channels in input tensor
    :param out_channels: number of channels in output tensor
    :param batch_norm: bool: use batch normalisation
    :return: torch.Tensor
    """
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        self.block = nn.Sequential()
        self.block.add_module('conv1', conv3x3(in_channels, out_channels))
        if batch_norm:
            self.block.add_module('bn1', nn.BatchNorm2d(out_channels))
        self.block.add_module('relu1', nn.ReLU())
        self.block.add_module('conv2', conv3x3(out_channels, out_channels))
        if batch_norm:
            self.block.add_module('bn2', nn.BatchNorm2d(out_channels))
        self.block.add_module('relu2', nn.ReLU())

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    """
    Creates for encoder U-net based architecture

    :param in_channels: number of channels in input tensor
    :param num_filters: int: number of output filters
    :param num_blocks: int:  number of blocks in encoder and decoder of the Unet implementation
    :param batch_norm: bool: use batch normalisation
    :return: list: outputs of the encoder
    """
    def __init__(self, in_channels, num_filters, num_blocks, batch_norm=False):
        super().__init__()

        self.num_blocks = num_blocks
        for i in range(num_blocks):
            in_channels = in_channels if not i else num_filters * 2 ** (i - 1)
            out_channels = num_filters * 2**i
            self.add_module(f'block{i + 1}', EncoderBlock(in_channels, out_channels))
            if i != num_blocks - 1:
                self.add_module(f'pool{i + 1}', nn.MaxPool2d(2, 2))

    def forward(self, x):
        acts = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f'block{i + 1}')(x)
            acts.append(x)
            if i != self.num_blocks - 1:
                x = self.__getattr__(f'pool{i + 1}')(x)
        return acts

class DecoderBlock(nn.Module):
    """
    Creates decoder block U-net based architecture

    :param out_channels: int: number of channels in output tensor
    :return: torch.Tensor
    """
    def __init__(self, out_channels):
        super().__init__()

        self.uppool = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upconv = conv3x3(out_channels * 2, out_channels)
        self.conv1 = conv3x3(out_channels * 2, out_channels)
        self.relu1= nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.relu2= nn.ReLU()

    def forward(self, down, left):
        x = self.uppool(down)
        x = self.upconv(x)
        x = torch.cat([left, x], 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class Decoder(nn.Module):
    """
    Creates decoder U-net based architecture

    :param num_filters: number of filters in the first layer in encoder
    :param num_blocks: number of blocks in encoder and decoder of the Unet implementation
    :return: torch.Tensor: shape of the tensor (width, height, nm_filters)
    """
    def __init__(self, num_filters, num_blocks):
        super().__init__()

        for i in range(num_blocks):
            self.add_module(f'block{num_blocks - i}', DecoderBlock(num_filters * 2**i))

    def forward(self, acts):
        up = acts[-1]
        for i, left in enumerate(acts[-2::-1]):
            up = self.__getattr__(f'block{i + 1}')(up, left)
        return up

class UNet(nn.Module):
    """
    Creates U-net graph

    :param num_blocks: number of blocks in encoder and decoder of the Unet implementation,
    be careful that shape of the input tensor divisible by 2**num_blocks
    :param num_filters:  number of filters in the first layer in encoder
    :param batch_norm: bool: use batch normalisation
    :return: torch.Tensor: output of the last layer
    """
    def __init__(self, num_classes=1, in_channels=1, num_filters=4, num_blocks=5, batch_norm=False):
        super().__init__()

        self.encoder = Encoder(in_channels, num_filters, num_blocks)
        self.decoder = Decoder(num_filters, num_blocks - 1)
        self.final = nn.Conv2d(num_filters, num_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)
        return x
