import torch
import torch.nn as nn

class ABeDice(nn.Module):
    def __init__(self, alpha=1, beta=0, n_classes=9):
        super(ABeDice, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta

    def _one_hot_encoder(self, input_tensor):
        if input_tensor.dim() == 4:  # [B,1,H,W]
            input_tensor = input_tensor.squeeze(1)

        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _ABeDice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, logits, target):
        p = torch.softmax(logits, dim=1)
        t = 1
        for i in range(self.beta):
            t = 1 - p ** t
        p = p ** (self.alpha * t)
        target = self._one_hot_encoder(target)
        assert p.size() == target.size(), 'predict {} & target {} shape do not match'.format(p.size(), target.size())

        loss = 0.0
        for i in range(1, self.n_classes):
            dice = self._ABeDice_loss(p[:, i], target[:, i])
            loss += dice
        return loss / (self.n_classes-1)