import cv2
import numpy as np
import torch
import numpy as np
from monai.metrics import compute_average_surface_distance
from sklearn.metrics import precision_score

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    mask: numpy array 2D, uint8, values {0,1}
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))

    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1:h + 1, 1:w + 1]
    return mask - mask_erode

# GitHub repo: https://github.com/bowenc0221/boundary-iou-api
def boundary_iou(y_true, y_pred, dilation_ratio=0.02):
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.5).float()
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
        y_pred = (y_pred == 1).float()

    y_true = (y_true == 1).float()

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    B = y_true.shape[0]
    y_true_np = y_true.squeeze(1).cpu().numpy().astype(np.uint8)  # (B, H, W)
    y_pred_np = y_pred.squeeze(1).cpu().numpy().astype(np.uint8)  # (B, H, W)

    ious = []
    for i in range(B):
        gt_boundary = mask_to_boundary(y_true_np[i], dilation_ratio)
        dt_boundary = mask_to_boundary(y_pred_np[i], dilation_ratio)

        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()

        if union == 0:
            # ambas máscaras (y por ende sus boundaries) vacías -> acierto trivial
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)
    return torch.tensor(ious, dtype=torch.float32, device=y_true.device)  # (B,)

def ASSD(y_true, y_pred):
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.5).float()
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
        y_pred = (y_pred == 1).float()

    y_true = (y_true == 1).float()  # siempre

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    assd_metric = compute_average_surface_distance(
        y_pred=y_pred, y=y_true,
        include_background=True, symmetric=True,
        distance_metric="euclidean") 

    return assd_metric.mean()

def precision(y_true, y_pred, threshold=0.5):
    # Activación
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

    # zero_division=0 evita error si no hay positivos predichos
    return precision_score(y_true_np, y_pred_np, zero_division=0)

def recall(y_true, y_pred, smooth=1e-4):
    # Activación
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred>0.5).float()
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        # argmax ANTES de slicear, sobre todos los canales
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)  # (B, 1, H, W) con índices de clase
        y_pred = (y_pred == 1).float()                       # máscara foreground (clase 1)
        y_true = (y_true == 1).float() 

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    y_true = y_true.float()
    TP = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    FN = torch.sum(y_true * (1 - y_pred), dim=[1, 2, 3])
    return torch.mean((TP + smooth) / (TP + FN + smooth), dim=0)

def dice_hard(y_true, y_pred, smooth = 1e-5):
    # Activación
    if y_pred.shape[1] == 1:
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred>0.5).float()
    else:
        y_pred = torch.softmax(y_pred, dim=1)
        # argmax ANTES de slicear, sobre todos los canales
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)  # (B, 1, H, W) con índices de clase
        y_pred = (y_pred == 1).float()                       # máscara foreground (clase 1)
        y_true = (y_true == 1).float() 

    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    y_true = y_true.float()

    intersection = torch.sum(y_true * y_pred, dim=[1,2,3])
    cardinality  = torch.sum(y_true + y_pred , dim=[1,2,3])
    return torch.mean((2. * intersection + smooth) / (cardinality + smooth), dim=0)