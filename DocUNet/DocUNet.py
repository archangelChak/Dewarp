from torch import nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from UNet.UNet import Encoder, Decoder
from utils.layers import conv3x3


def train_epoch(model, optimizer, train_loader, criterion, device):
    """
    One epoch of training for model

    :param train_loader: torch.utils.data.DataLoader: DataLoader for input training data (images,masks)
    :param model: torch.nn.Module: model
    :param criterion: loss
    :param optimizer: optimizer for model training
    :param device: device on which computation is executed
    """

    model.train()
    
    for batch_train, batch_answers in train_loader:
        batch_train = batch_train.to(device)
        batch_answers = batch_answers.to(device)
        
        optimizer.zero_grad()
        
        model_answers = model(batch_train)
        
        new_loss = criterion(model_answers, batch_answers)
        new_loss.backward()
        optimizer.step()


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

    def __init__(self, num_classes=2, in_channels=1, num_filters=4, num_blocks=5):
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
        y = self.final(x)
        x = self.encoder1(x)
        x = self.decoder1(x)
        x = self.final(x)
        return y,x
