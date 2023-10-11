import torch
import torch.nn
import torch.nn as nn
import numpy as np


class SparseBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self):
        # super().__init__(reduction="none")
        super().__init__()

    def forward(self, predicted, target):
        # return super().forward(predicted, target.to(torch.float16))
        b, c, _, _ = predicted.shape
        loss = super().forward(predicted, target.to(torch.float16))
        # print(b, c, max(1, ((b * c) - torch.sum((torch.sum(target, dim=(-2, -1)) != 0)))))
        return (
            loss
            * torch.sum((torch.sum(target, dim=(-2, -1)) != 0))
            / max(1, ((b * c) - torch.sum((torch.sum(target, dim=(-2, -1)) != 0))))
        )

        loss = torch.mean(loss, dim=(-2, -1))
        return torch.mean(loss[:, 10])
        pred = torch.sigmoid(predicted) > 0.5
        index = (torch.sum(pred, dim=(-2, -1)) != 0) | (torch.sum(target, dim=(-2, -1)) != 0)
        # print(torch.sum(index))
        return torch.sum(loss) / torch.sum(index)
