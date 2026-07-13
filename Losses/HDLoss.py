import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from Losses.Common.DiceForeground import DiceForeground

# -----------------------------------------------------------------------------------
# Hausdorff Distance Loss (HDLoss) similar to MONAI implementation (HausdorffDTLoss)
# -----------------------------------------------------------------------------------
class HDLoss(nn.Module):
    """
    Karimi, D. et. al. (2019)
    """
    def __init__(self, alpha=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    @torch.no_grad()
    def distance_field(self, mask_np: np.ndarray) -> np.ndarray:
        field = np.zeros_like(mask_np, dtype=np.float32)
        for b in range(mask_np.shape[0]):
            fg_mask = mask_np[b] > 0.5
            if fg_mask.any() and not fg_mask.all():
                fg_dist = distance_transform_edt(fg_mask)
                bg_dist = distance_transform_edt(~fg_mask)
                combined = fg_dist + bg_dist
                # Normalized map distance
                max_val = combined.max()
                if max_val > 0:
                    field[b] = combined / max_val
        return field

    def forward(self, logits, target):
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.float()

        probs = F.softmax(logits, dim=1)
        pred = probs[:, 1]  # foreground, [B,H,W]

        pred_np = (pred.detach() > 0.5).float().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        pred_dt = torch.from_numpy(self.distance_field(pred_np)).to(pred.device)
        target_dt = torch.from_numpy(self.distance_field(target_np)).to(pred.device)

        pred_error = (pred - target) ** 2
        distance = pred_dt.pow(self.alpha) + target_dt.pow(self.alpha)

        dt_field = pred_error * distance

        if self.reduction == "mean":
            return dt_field.mean()
        elif self.reduction == "sum":
            return dt_field.sum()
        return dt_field
    

# --------------
# DICE + HD Loss
# --------------
class DiceHDLoss(nn.Module):
    def __init__(self, dice_weight=0.5, hd_weight=0.5, alpha=2.0, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.hd_weight = hd_weight
        self.dice = DiceForeground(smooth=smooth)
        self.hd = HDLoss(alpha=alpha)

    def forward(self, logits, target):
        return self.dice_weight * self.dice(logits, target) + self.hd_weight * self.hd(logits, target)