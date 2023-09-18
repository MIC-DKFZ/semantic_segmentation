import torch
import torchmetrics
from src.metric.confmat import ConfusionMatrix


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


# class IoU_MultiLabel(torchmetrics.classification.confusion_matrix.MultilabelConfusionMatrix):
class IoU_MultiLabel(torchmetrics.classification.confusion_matrix.MultilabelConfusionMatrix):
    def __init__(self, num_classes, labels, name=" ", per_class=True, **kwargs):
        self.labels = labels
        self.name = name
        self.per_class = per_class
        # super().__init__(task="multilabel", num_labels=num_classes, **kwargs)
        super().__init__(num_labels=num_classes, **kwargs)

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
