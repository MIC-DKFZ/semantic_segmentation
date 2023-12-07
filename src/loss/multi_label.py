import torch
import torch.nn
import torch.nn as nn
import numpy as np
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss
from segmentation_models_pytorch.losses.constants import (
    BINARY_MODE,
    MULTICLASS_MODE,
    MULTILABEL_MODE,
)
import torch.nn.functional as F
from segmentation_models_pytorch.losses._functional import soft_jaccard_score, to_tensor


class MLJaccardLoss(JaccardLoss):
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        super().__init__(mode="multilabel")
        #

    def forward(self, y_pred, y_true):
        # Create a mask for ignoring the index
        mask = y_true != self.ignore_index
        y_pred = y_pred * mask.type_as(y_pred)  # Apply mask to predictions
        y_true = y_true * mask.type_as(y_true)  # Apply mask to targets
        return super().forward(y_pred, y_true)


class MLBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        super().__init__(reduction="none")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mask = y_true != self.ignore_index
        # Compute the loss for the labelled data only
        return F.binary_cross_entropy_with_logits(y_pred[mask], y_true[mask].float())


class ML_BCE_JL_Loss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.jaccard_loss = MLJaccardLoss(ignore_index)
        self.bce_loss = MLBCEWithLogitsLoss(ignore_index)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        jl = self.jaccard_loss(y_pred, y_true)
        bc = self.bce_loss(y_pred, y_true)
        return (jl + bc) / 2


class JaccardLossPL(JaccardLoss):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, map) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = 2

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_jaccard_score(
            y_pred,
            y_true.type(y_pred.dtype),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.float()
        # print(loss.mean(), loss[map].mean())
        if self.classes is not None:
            loss = loss[self.classes]
        # return loss[map].mean() * map.numel() / map.sum()
        return loss[map].mean() * map.sum() / map.numel()
        # return loss.mean()


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
