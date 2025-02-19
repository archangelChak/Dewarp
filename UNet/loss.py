import torch


def dice_loss(pred_tensor, target_tensor):
    """
    Evaluates dice loss for two vectors

    :param pred_tensor: torch.Tensor: predictions (batch_size, n_features)
    :param target_tensor: torch.Tensor: originals (batch_size, n_features)
    :return: dice_loss
    """

    smooth = 1.
    iflat = pred_tensor.contiguous().view(-1)
    tflat = target_tensor.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)
    
    return ((2. * intersection + smooth) / (A_sum + B_sum + smooth)) 


def doc_loss(pred_tensor, target_tensor):
    """
    Evaluates custom DocLoss for two vectors

    :param pred_tensor: torch.Tensor: predictions 
    :param target_tensor: torch.Tensor: originals
    :return: float: custom loss
    """

    target_foreground = target_tensor[target_tensor>0]
    pred_foreground = pred_tensor[target_tensor>0]
    pred_background = pred_tensor[target_tensor<0]

    foreground_error = pred_foreground - target_foreground
    n2 = pred_background.numel()
    foreground_error_final = torch.abs(foreground_error).mean() + torch.abs(foreground_error.mean())*0.1
    background_error = pred_background[pred_background>0]
    background_error_final = background_error.sum()/(n2+1)
    return background_error_final + foreground_error_final

def evaluate_loss(loader, model, criterion, device):
    """
    Evaluates average loss for model for one epoch

    :param loader: torch.utils.data.DataLoader: DataLoader for input data (images,masks)
    :param model: torch.nn.Module: model
    :param criterion: loss
    :param device: device on which computation is executed
    :return: float: average loss
    """

    model.eval()
    
    total_loss = 0
    total_dice = 0
    total_n = 0
    
    with torch.no_grad():
        for batch_test, batch_answers in loader:
            batch_test = batch_test.to(device)
            batch_answers = batch_answers.to(device)
            
            model_answers = model(batch_test)[1]
            one_batch_loss = float(criterion(model_answers, batch_answers))            
            total_loss += one_batch_loss
            total_n += 1
    
    return total_loss / total_n


def evaluate_loss_with_dice(loader, model, criterion, device):
    """
    Evaluates average loss and dice loss for model for one epoch

    :param loader: torch.utils.data.DataLoader: DataLoader for input data (images,masks)
    :param model: torch.nn.Module: model
    :param criterion: loss
    :param device: device on which computation is executed
    :return: float: average loss, average dice loss
    """

    model.eval()
    
    total_loss = 0
    total_dice = 0
    total_n = 0
    
    with torch.no_grad():
        for batch_test, batch_answers in loader:
            batch_test = batch_test.to(device)
            batch_answers = batch_answers.to(device)
            
            model_answers = torch.sigmoid(model(batch_test))
            one_batch_loss = float(criterion(model_answers[:,0], batch_answers))
            one_batch_dice = float(dice_loss(model_answers[:,0], batch_answers))
            
            total_loss += one_batch_loss
            total_dice += one_batch_dice
            total_n += 1
    
    return (total_loss / total_n, total_dice / total_n)
