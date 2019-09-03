import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        
        new_loss = criterion(model_answers[:,0], batch_answers)
        new_loss.backward()
        optimizer.step()

def TrainNet(model, ellipses_train_loader,ellipses_test_loader,number_of_epochs=100):
    writer = SummaryWriter()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()



    for epoch in tqdm(range(0, number_of_epochs)):
        train_epoch(model, optimizer=optimizer,
                    train_loader=ellipses_train_loader,
                    criterion=criterion, device='cuda') 
        train_loss, train_dice = evaluate_loss(loader=ellipses_train_loader, model=model,
                                               criterion=criterion, device='cuda')
        val_loss, val_dice = evaluate_loss(loader=ellipses_test_loader, model=model,
                                           criterion=criterion,
                                           device='cuda')
        writer.add_scalar('data/train_logloss', train_loss, epoch)
        writer.add_scalar('data/train_dice', train_dice, epoch)
        writer.add_scalar('data/test_logloss', val_loss, epoch)
        writer.add_scalar('data/test_dice', val_dice, epoch)

        preds = torch.sigmoid(model(image.to('cuda'))).cpu()

        writer.add_images('preds', preds, epoch)
    return model