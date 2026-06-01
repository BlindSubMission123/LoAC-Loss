import torch
import torch.nn.functional as F
import torch.nn as nn

class NAC_LossS(nn.Module):
    def __init__(self, NUM_CLASS=2, m=1e-4, smooth = 1e-6):
        super().__init__()
        self.m = m
        self.smooth = smooth
        self.NUM_CLASS = NUM_CLASS

    def RegionTerm(self, y_true, y_pred):
        dim = (1,2,3)
        yTrueOnehot = torch.zeros_like(y_pred)
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)[:,1:]
        y_pred = y_pred[:,1:]

        num = y_pred*((1-yTrueOnehot)**2) + (1-y_pred)*((yTrueOnehot)**2)
        loss = torch.sum(num, dim = dim) / torch.sum(yTrueOnehot*yTrueOnehot + y_pred*y_pred - yTrueOnehot *y_pred + self.smooth, dim = dim)
        return torch.mean(loss)

    def GradientLoss(self, y_pred, penalty="l1"):
        dH = torch.abs(y_pred[..., 1:] - y_pred[..., :-1])
        dW = torch.abs(y_pred[:, :, 1:] - y_pred[:, :, :-1])
        if penalty == "l2":
            dH = dH * dH
            dW = dW * dW
        loss = torch.sum(dH) + torch.sum(dW)
        return loss

    def forward(self, y_pred, y_true):
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        _,_,H,W = y_pred.shape
        # Soft de la pred
        y_pred = torch.softmax(y_pred,dim=1)
        
        region = self.RegionTerm(y_true, y_pred)
        length = self.GradientLoss(y_pred)
        return region + self.m *(1/(H*W))* length