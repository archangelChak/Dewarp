import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from UNet.loss import evaluate_loss_with_dice


def train_epoch(model, optimizer, train_loader, criterion, device):
    """
    One epoch of training for model

    :param train_loader: torch.utils.data.DataLoader: DataLoader for input training data (images,masks)
    :param model: torch.nn.Module: model
    :param criterion: loss
    :param optimizer: optimizer for model training
    :param device: device on which computation is executed
    :param device: device to compute on
    """

    model.train()
    
    for batch_train, batch_answers in train_loader:
        batch_train = batch_train.to(device)
        batch_answers = batch_answers.to(device)
        
        optimizer.zero_grad()
        
        model_answers = model(batch_train)
        
        new_loss = criterion(model_answers[:,0], batch_answers)
        new_loss.backward()
        optimizer.step()


def train_net(model, train_loader,test_loader,number_of_epochs=100, device = 'cuda'):
    """
    Training UNet
    param: model: model to train
    param: train_loader: torch.utils.data.DataLoader: train dataloader
    param: test_loader: torch.utils.data.DataLoader: test dataloader
    param: number_of_epochs: int: number of epochs
    return: model: trained model
    """

    writer = SummaryWriter()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in tqdm(range(0, number_of_epochs)):
        train_epoch(model, optimizer=optimizer,
                    train_loader=train_loader,
                    criterion=criterion, device=device ) 
        train_loss, train_dice = evaluate_loss_with_dice(loader=train_loader, model=model,
                                               criterion=criterion, device=device )
        val_loss, val_dice = evaluate_loss_with_dice(loader=test_loader, model=model,
                                           criterion=criterion,
                                           device=device )
        writer.add_scalar('data/train_logloss', train_loss, epoch)
        writer.add_scalar('data/train_dice', train_dice, epoch)
        writer.add_scalar('data/test_logloss', val_loss, epoch)
        writer.add_scalar('data/test_dice', val_dice, epoch)
    return model
