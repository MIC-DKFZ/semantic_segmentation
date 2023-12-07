from typing import Any

import torch
import torchmetrics
from src.metric.confmat import ConfusionMatrix
import torch.nn.functional as F


class IoU(ConfusionMatrix):
    def __init__(
        self,
        per_class: bool = False,
        name: str = "IoU",
        replace_nan: bool = True,
        ignore_bg: bool = False,
        **kwargs,
    ):
        """
        Init the IoU Class as subclass of ConfusionMatrix

        Parameters
        ----------
        per_class : bool, optional
            If False the mean IoU over the classes is returned
            If True additionally the IoU for each class is returned
        name : str, optional
            Name of the metric, used as prefix for logging the class scores
        replace_nan : bool, optional
            replace NaN by 0.0
        kwargs:
            arguments passed to the super class (ConfusionMatrix)
        """

        super().__init__(**kwargs)

        self.per_class = per_class
        self.name = name
        self.replace_nan = replace_nan
        self.ignore_bg = ignore_bg

    def get_iou_from_mat(self, confmat: torch.Tensor) -> torch.Tensor:
        """
        Computing the IoU from a confusion matrix (class wise)

        Parameters
        ----------
        confmat : torch.Tensor

        Returns
        -------
        torch.Tensor :
            Tensor contains the IoU for each class
        """
        intersection = confmat.diag()
        IoU = intersection / (confmat.sum(1) + confmat.sum(0) - intersection)

        # for using a ignore class
        # if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
        #    IoU = torch.cat((IoU[: self.ignore_class], IoU[self.ignore_class + 1 :]))

        return IoU

    def compute(self) -> dict or torch.Tensor:
        """
        Compute the IoU from the confusion matrix.
        Depended on initialization return mean IoU and IoU per class or only mean IoU.

        Returns
        -------
        dict or torch.Tensor :
            a single Tensor if only mean IoU is returned, a dict if additionally the class wise
            IoU is returned
        """
        IoU = self.get_iou_from_mat(self.mat.clone())
        # if self.replace_nan:
        #     IoU[IoU.isnan()] = 0.0

        if self.ignore_bg:
            mIoU = torch.nanmean(IoU[1:])  # .mean()
        else:
            mIoU = torch.nanmean(IoU)  # .mean()

        if self.per_class:
            IoU = {self.name + "_" + self.labels[i]: IoU[i] for i in range(len(IoU))}
            IoU["mean" + self.name] = mIoU
            return IoU
        else:
            return mIoU


from torchmetrics.classification.confusion_matrix import MultilabelConfusionMatrix
from torchmetrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_arg_validation,
    _binary_confusion_matrix_compute,
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _binary_confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_compute,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_compute,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from torchmetrics.utilities.data import _bincount
from torch import Tensor


def multilabel_confusion_matrix_update(preds: Tensor, target: Tensor, num_labels: int) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = (
        (2 * target + preds) + 4 * torch.arange(num_labels, device=preds.device)
    ).flatten()
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = _bincount(unique_mapping, minlength=4 * num_labels)
    print("BB", bins.shape)
    return bins.reshape(num_labels, 2, 2)
    # ignore = (target > 1).flatten()
    #
    # unique_mapping = (
    #     (2 * target + preds) + 4 * torch.arange(num_labels, device=preds.device)
    # ).flatten()
    # unique_mapping = unique_mapping[~ignore]
    # unique_mapping = unique_mapping[unique_mapping >= 0]
    # bins = _bincount(unique_mapping, minlength=4 * num_labels)
    # return bins.reshape(num_labels, 2, 2)


# class IoU_MultiLabel(torchmetrics.classification.confusion_matrix.MultilabelConfusionMatrix):
class IoU_MultiLabel(torchmetrics.classification.confusion_matrix.MultilabelConfusionMatrix):
    def __init__(self, num_classes, labels, name=" ", per_class=True, **kwargs):
        self.labels = labels
        self.name = name
        self.per_class = per_class
        # super().__init__(task="multilabel", num_labels=num_classes, **kwargs)
        super().__init__(num_labels=num_classes, **kwargs)

    def update(self, preds, target, *args: Any, **kwargs: Any) -> Any:
        # print(self.ignore_index, self.num_labels, preds.shape, target.shape, *args, **kwargs)
        # print(torch.unique(target))
        # print(torch.unique(preds))
        # tensor([0, 1, 255])
        # tensor([-4.6647, -4.5634, -4.5100, ..., 4.3468, 4.3824, 4.6403])
        # 255 16 torch.Size([6, 16, 345, 1024]) torch.Size([6, 16, 345, 1024])
        # 255 16 torch.Size([6, 16, 100, 100]) torch.Size([6, 16, 100, 100])
        # pred = torch.sigmoid(pred)
        # pred = F.logsigmoid(pred).exp()
        return super().update(
            preds,
            target.long(),
        )  # *args, **kwargs)

        # """Update state with predictions and targets."""
        # if self.validate_args:
        #     _multilabel_confusion_matrix_tensor_validation(
        #         preds, target, self.num_labels, self.ignore_index
        #     )
        # preds, target = _multilabel_confusion_matrix_format(
        #     preds, target, self.num_labels, self.threshold, self.ignore_index
        # )
        # print(preds.shape, target.shape)
        # confmat = multilabel_confusion_matrix_update(preds, target, self.num_labels)
        # # print(confmat.shape)
        # # print(torch.unique(confmat))
        # self.confmat += confmat

    def compute(self):
        tp = self.confmat[:, 1, 1]
        # tn = self.confmat[:, 0, 0]
        fp = self.confmat[:, 0, 1]
        fn = self.confmat[:, 1, 0]
        IoU = tp / (tp + fp + fn)
        mIoU = torch.nanmean(IoU)

        if self.per_class:
            IoU = {self.name + "_" + self.labels[i]: IoU[i] for i in range(len(IoU))}
            IoU["mean" + self.name] = mIoU
            return IoU
        else:
            return mIoU
