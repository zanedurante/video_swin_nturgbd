import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss()
    def forward(self, anchor, positive, negative):
        loss = self.triplet_loss(anchor, positive, negative)
        return loss