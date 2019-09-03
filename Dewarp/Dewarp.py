import torch.optim as optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from UNet.Loss import DocLoss
from torchvision import transforms
from Dataset.dataloader import GetLoader
from Processing.PreProcessing import Smooth
from Processing.PostProcessing import Find_Grid, Find_Rect, Create_Image, PixelRemap
from DocUNet import DocUNet
from PIL import Image, ImageDraw
from tqdm import tqdm
from torch import nn
import torch
import torch.optim as optim

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
    :return: float: average loss
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


def make_docunet_estimator(model,train_loader, test_loader, num_of_epochs=10):
    """
    Training and evaluation for DocUnet model

    :param num_of_epochs: int: number of epochs for training
    :param train_loader: torch.utils.data.DataLoader: dataLoader of input data for training
    :param test_loader: torch.utils.data.DataLoader: dataLoader of input data for testing
    :return: torch.nn.Module: trained model
    """
    writer = SummaryWriter()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DocLoss

    for epoch in tqdm(range(0, num_of_epochs)):
        train_epoch(model, optimizer=optimizer,
                   train_loader=train_loader,
                   criterion=criterion, device='cuda') 
        train_loss = evaluate_loss(loader=train_loader, model=model,
                                           criterion=criterion, device='cuda')
        #val_loss = evaluate_loss(loader=test_loader, model=model,
        #                                  criterion=criterion, device='cuda')
        writer.add_scalar('data/train_loss', train_loss, epoch)
    return model
def ShowResult(test_image, pred_mask):
    """
    Showing dewarped images of test_image (using grid and using pixel to pixel remap)

    :param test_image: PIL.Image: input image
    :param pred_mask: numpy.ndarray: predicted mask
    :return: tuple(PIL.Image, PIL.Image): tuple of dewarped images
    """
    if (type(test_image) == torch.Tensor):
        test_image = transforms.ToPILImage()(test_image)
    else:
        test_image = Image.fromarray(test_image)
    new_mask = pred_mask
    grid = Find_Grid(Smooth(new_mask),new_mask)
    rectangles = Find_Rect(grid,np.asarray(test_image))
    grid_dewarped = Create_Image(rectangles)
    pixel_dewarped = Image.fromarray(PixelRemap(np.asarray(test_image),new_mask))
    return (grid_dewarped, pixel_dewarped)
def GetPredMask(test_image, model):
    """
    Getting predicted mask for test_image (using grid and using pixel to pixel remap)

    :param test_image: PIL.Image: input image
    :param model: nn.Module: trained model
    :return: tuple(PIL.Image, PIL.Image): tuple of dewarped images
    """
    test_image= test_image=transforms.ToPILImage()(test_image)
    #test_image=Image.fromarray(test_image)
    new_mask = model(transforms.ToTensor()(test_image).unsqueeze(1).cuda()).transpose(1,2).transpose(2,3).cpu().detach().numpy().squeeze()
    return new_mask








