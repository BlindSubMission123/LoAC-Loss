from Losses.Common.CrossEntropy import *
from Losses.Common.DiceForeground import *

import torch.nn as nn
import torch.nn.functional as F


class CEDice(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()

        self.ce = CrossEntropySeg()
        self.dice = DiceForeground(smooth)

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):

        ce = self.ce(logits, target)

        dice = self.dice(logits, target)

        loss = (self.ce_weight * ce + self.dice_weight * dice)
        return loss