import torch
import numpy as np
from sklearn.metrics import precision_score
from monai.metrics import HausdorffDistanceMetric

def dice_soft(y_true, y_pred, smooth=1e-6):
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = y_pred[:, 1:2, :, :]  

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    y_true = (y_true == 1).float()  
    
    intersection = torch.sum(y_true * y_pred, dim=(1,2,3))
    cardinality  = torch.sum(y_true + y_pred, dim=(1,2,3))

    dice = (2. * intersection + smooth) / (cardinality + smooth)
    return dice.mean()

def dice_hard(y_true, y_pred, smooth = 1e-5):
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred>0.5).float()
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
        y_pred = (y_pred == 1).float()                   
        y_true = (y_true == 1).float() 

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    y_true = y_true.float()

    intersection = torch.sum(y_true * y_pred, dim=[1,2,3])
    cardinality  = torch.sum(y_true + y_pred , dim=[1,2,3])
    return torch.mean((2. * intersection + smooth) / (cardinality + smooth), dim=0)

def HD_distance(y_true, y_pred):
    # Activación
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred>0.5).float()
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True) 
        y_pred = (y_pred == 1).float()                     
        y_true = (y_true == 1).float() 

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    y_true = y_true.float()

    hd_metric = HausdorffDistanceMetric(percentile=95)
    hd_metric(y_pred, y_true)          
    result = hd_metric.aggregate()      
    hd_metric.reset()                   
    return result

def precision(y_true, y_pred, smooth=1e-4):
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred>0.5).float()
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)  
        y_pred = (y_pred == 1).float()                      
        y_true = (y_true == 1).float() 

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    y_true = y_true.float()

    TP = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    FP = torch.sum((1 - y_true) * y_pred, dim=[1, 2, 3])
    return torch.mean((TP + smooth) / (TP + FP + smooth), dim=0)

def recall(y_true, y_pred, smooth=1e-4):
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred>0.5).float()
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)  
        y_pred = (y_pred == 1).float()                     
        y_true = (y_true == 1).float() 

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    y_true = y_true.float()
    TP = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    FN = torch.sum(y_true * (1 - y_pred), dim=[1, 2, 3])
    return torch.mean((TP + smooth) / (TP + FN + smooth), dim=0)

def iou(y_true, y_pred, threshold=0.5, eps=1e-7):
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = y_pred[:, 1:2, :, :]  # foreground

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    y_true = y_true.float()
    y_pred = (y_pred > threshold).float()

    y_true_np = y_true.detach().cpu().numpy().ravel()
    y_pred_np = y_pred.detach().cpu().numpy().ravel()

    intersection = np.sum(y_true_np * y_pred_np)
    union = np.sum(y_true_np) + np.sum(y_pred_np) - intersection
    return (intersection + eps) / (union + eps)


def precision(y_true, y_pred, threshold=0.5):
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = y_pred[:, 1:2, :, :]  # foreground

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    # Binarización
    y_true = y_true.float()
    y_pred = (y_pred > threshold).float()

    y_true_np = y_true.detach().cpu().numpy().ravel()
    y_pred_np = y_pred.detach().cpu().numpy().ravel()
    return precision_score(y_true_np, y_pred_np, zero_division=0)