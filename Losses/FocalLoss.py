import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.long()

        logp = F.log_softmax(logits, dim=1)          # [B,2,H,W]
        p = logp.exp()

        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)  # [B,H,W]
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)        # [B,H,W]

        alpha_t = torch.where(
            target == 1,
            torch.full_like(p_t, self.alpha),
            torch.full_like(p_t, 1 - self.alpha),
        )

        loss = -alpha_t * (1 - p_t).pow(self.gamma) * logp_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss