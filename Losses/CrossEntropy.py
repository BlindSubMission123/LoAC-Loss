import torch
import torch.nn as nn

class CrossEntropySeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):

        if target.dim() == 4:
            target = target.squeeze(1)

        loss = self.ce(logits,target.long())

        return loss