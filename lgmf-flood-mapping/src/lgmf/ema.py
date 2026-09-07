"""Exponential moving average of model weights.

Keep a shadow copy that trails the trained weights; evaluate and checkpoint the
shadow. Cheap, and reliably worth ~0.5-1 IoU while smoothing the val curve.
"""
import copy
import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            m = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                v.copy_(m)               # counters / int buffers: just track
