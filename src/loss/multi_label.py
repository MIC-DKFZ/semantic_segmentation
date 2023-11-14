import torch
import torch.nn
import torch.nn as nn
import numpy as np


class SparseBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self):
        super().__init__(reduction="none")
        # super().__init__()

    def forward(self, predicted, target):
        # return super().forward(predicted, target.to(torch.float16))
        b, c, _, _ = predicted.shape
        loss = super().forward(predicted, target.to(torch.float16))
        loss = torch.mean(loss, dim=(-2, -1))
        index = torch.sum(target, dim=(-2, -1)) > 0
        index2 = torch.sum(torch.sigmoid(predicted) > 0.5, dim=(-2, -1)) > 0
        loss = torch.sum(loss) / torch.sum(index + index2)
        # print(loss)
        # print("1", index)
        # print("2", index2)
        # print("3", index + index2)
        # print("3", torch.sum(index + index2))
        return loss

        loss = torch.mean(loss, dim=(-2, -1))
        return torch.mean(loss[:, 10])
        pred = torch.sigmoid(predicted) > 0.5
        index = (torch.sum(pred, dim=(-2, -1)) != 0) | (torch.sum(target, dim=(-2, -1)) != 0)
        # print(torch.sum(index))
        return torch.sum(loss) / torch.sum(index)
