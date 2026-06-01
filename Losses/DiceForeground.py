import torch.nn as nn
import torch.nn.functional as F

class DiceForeground(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: [B,2,H,W]
        target: [B,H,W] o [B,1,H,W]
        """

        if target.dim() == 4:
            target = target.squeeze(1)

        target = target.float()
        probs = F.softmax(logits, dim=1)

        # Solo foreground
        pred = probs[:, 1]
        intersection = (pred * target).sum(dim=(1,2))
        union = (pred.sum(dim=(1,2)) + target.sum(dim=(1,2)))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()