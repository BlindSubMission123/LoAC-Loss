import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from Losses.Common.DiceForeground import DiceForeground

class BoundaryLoss(nn.Module):
    """
    Boundary Loss (Kervadec et al., 2019 - "Boundary loss for highly
    unbalanced segmentation")
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    @torch.no_grad()
    def compute_sdf(self, mask_np: np.ndarray) -> np.ndarray:
        sdf = np.zeros_like(mask_np, dtype=np.float32)
        for b in range(mask_np.shape[0]):
            posmask = mask_np[b].astype(bool)
            if posmask.any() and not posmask.all():
                negmask = ~posmask
                pos_dist = distance_transform_edt(posmask)
                neg_dist = distance_transform_edt(negmask)
                sdf[b] = neg_dist * negmask - (pos_dist - 1) * posmask

                # Normalized
                max_abs = np.max(np.abs(sdf))
                if max_abs > 0:
                    sdf = sdf / max_abs
            # máscara toda 0 o toda 1 -> sdf queda en 0
        return sdf

    def forward(self, logits, target):
        if target.dim() == 4:
            target = target.squeeze(1)

        target = target.float()
        probs = F.softmax(logits, dim=1)

        pred = probs[:, 1]  # foreground, con gradiente
        target_np = target.detach().cpu().numpy()
        sdf = torch.from_numpy(self.compute_sdf(target_np)).to(pred.device)

        multiplied = pred * sdf

        if self.reduction == "mean":
            return multiplied.mean()
        elif self.reduction == "sum":
            return multiplied.sum()
        
        return multiplied
    
# ---------------------
# DICE + Boundary Loss
# ---------------------
class DiceBoundaryLoss(nn.Module):
    def __init__(self, dice_weight=0.5, boundary_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.dice = DiceForeground(smooth=smooth)
        self.boundary = BoundaryLoss()

    def forward(self, logits, target):
        return self.dice_weight * self.dice(logits, target) + self.boundary_weight * self.boundary(logits, target)