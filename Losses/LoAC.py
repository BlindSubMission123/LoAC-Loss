import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class LoAC(nn.Module):
    def __init__(self, n_classes=2, mu=1, alpha=1):
        super(LoAC, self).__init__()
        # Classes and mu cte
        self.n_classes = n_classes

        # Suavidad raiz
        self.eps = 1e-6

        # Constants
        self.mu = mu
        self.alpha = alpha

    def Estimate_FRWD_diff(self, x):
        # Forward difference
        gx = x[:, :, 1:, :] - x[:, :, :-1, :]
        gy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return gx, gy

    def Estimate_DIAG_diff(self, x):
        # Diagonal difference
        cte = 2**0.5
        gx = (x[:, :, :-1, 1:] - x[:, :, 1:, :-1]) / cte
        gy = (x[:, :, 1:, 1:] - x[:, :, :-1, :-1]) / cte
        return gx, gy
        
    def Estimate_CEN_diff(self, x):
        # Center difference
        gx = (x[:, :, 1:-1, 2:] - x[:, :, 1:-1, :-2]) / 2
        gy = (x[:, :, 2:, 1:-1] - x[:, :, :-2, 1:-1]) / 2
        return gx, gy
    
    def RegionTerm2(self, Prediction, Label, k=1):
        dim = (1,2,3)
        p_k   = Prediction[:, k:k+1]     # (B, 1, H, W)
        lbl_k = Label[:, k:k+1]    # (B, 1, H, W)

        num = torch.sum(p_k*((1-lbl_k)**2) + (1-p_k)*((lbl_k)**2),dim=dim)
        # Self.eps inside den as IoU loss implemenation
        den = torch.sum(lbl_k*lbl_k + p_k*p_k - lbl_k *p_k + self.eps, dim = dim)
        return num/den
    
    def OrientationTerm(self, gx_p, gy_p, gx_l, gy_l):
        # Cosine similarity
        dot = gx_p * gx_l + gy_p * gy_l
        mag_label_sq = gx_l**2 + gy_l**2

        mag_pred  = torch.sqrt(gx_p**2 + gy_p**2 + self.eps)
        mag_label = torch.sqrt(gx_l**2 + gy_l**2 + self.eps)
        cos_sim = dot / (mag_pred * mag_label)

        # Similiraty normalized through only true border
        orient_loss = ((1.0 - cos_sim) * mag_label_sq)
        # Normalized similarity
        normalizer = mag_label_sq.sum(dim=(1,2,3)).clamp(min=self.eps)
        return orient_loss.sum(dim=(1,2,3)) / normalizer
    
    def MagnitudTerm(self, gx_pred, gy_pred, gx_lbl, gy_lbl):
        # Only magnitud information
        gx_pred = torch.abs(gx_pred)
        gy_pred = torch.abs(gy_pred)
        gx_lbl = torch.abs(gx_lbl)
        gy_lbl = torch.abs(gy_lbl)

        # Normalized magnitud
        num = (gx_pred - gx_lbl).pow(2).sum(dim=(1,2,3)) + (gy_pred - gy_lbl).pow(2).sum(dim=(1,2,3))
        den = (gx_pred.pow(2) + gx_lbl.pow(2)).sum(dim=(1,2,3)) + (gy_pred.pow(2) + gy_lbl.pow(2)).sum(dim=(1,2,3)) + self.eps
        return num / den
    
    def LengthOptimized(self, Prediction, Label, k):
        p_k   = Prediction[:, k:k+1]     # (B, 1, H, W)
        lbl_k = Label[:, k:k+1] 

        # Forward
        gx_pred, gy_pred = self.Estimate_FRWD_diff(p_k)
        gx_lbl, gy_lbl = self.Estimate_FRWD_diff(lbl_k)
        # Diagonal
        gx_pred_diag, gy_pred_diag = self.Estimate_DIAG_diff(p_k)
        gx_lbl_diag, gy_lbl_diag = self.Estimate_DIAG_diff(lbl_k)
        # Center
        gx_pred_cen, gy_pred_cen = self.Estimate_CEN_diff(p_k)
        gx_lbl_cen, gy_lbl_cen = self.Estimate_CEN_diff(lbl_k)

        # Magnitud
        L_FWRD = self.MagnitudTerm(gx_pred, gy_pred, gx_lbl, gy_lbl)
        L_DIAG = self.MagnitudTerm(gx_pred_diag, gy_pred_diag, gx_lbl_diag, gy_lbl_diag)
        L_CEN = self.MagnitudTerm(gx_pred_cen, gy_pred_cen, gx_lbl_cen, gy_lbl_cen)
        L_MAG = (L_FWRD + L_DIAG + L_CEN) /3

        # Orentation
        L_CEN_O = self.OrientationTerm(gx_pred_cen, gy_pred_cen, gx_lbl_cen, gy_lbl_cen)
        return (L_MAG + self.alpha*L_CEN_O)
    
    def forward(self, logits, label):
        # logits: (B, C, H, W)
        B, C, H, W = logits.size()

        # Softmax en lugar de sigmoid: (B, C, H, W)
        predication = torch.softmax(logits, dim=1)         

        # One-hot: (B, H, W) → (B, C, H, W)
        label_onehot = torch.zeros_like(predication)
        # Scatter: # (B, C, H, W)
        label_onehot.scatter_(1, label.long().unsqueeze(1), 1)

        total_length = torch.zeros(B, device=logits.device)
        total_region = torch.zeros(B, device=logits.device)

        # Only foreground
        for k in range(1,self.n_classes):
            total_length += self.LengthOptimized(predication, label_onehot, k)
            total_region += self.RegionTerm2(predication, label_onehot, k)

        # Region + Length
        return torch.mean(total_region + self.mu*total_length) / (self.n_classes - 1)