import numpy as np
import hydra
from omegaconf import DictConfig
import torch
from torchmetrics import Metric, MetricCollection, F1Score
from torchmetrics.utilities.data import dim_zero_cat
import pytorch_lightning as pl
from torchmetrics.utilities.data import _bincount
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.utils import get_logger

log = get_logger(__name__)


class MetricModule(MetricCollection):
    def __init__(self, config: DictConfig, **kwargs) -> None:
        """
        Init MetricModule as subclass of MetricCollection
        Can directly be initialized with the config

        Parameters
        ----------
        config: DictConfig
        kwargs :
            arguments passed to super class (MetricCollection)
        """

        metrics = {}
        for name, m_conf in config.items():
            if m_conf is not None:
                metric = hydra.utils.instantiate(m_conf)
                metrics[name] = metric
        super().__init__(metrics, **kwargs)


class MetricModule_AGGC(MetricCollection):
    def __init__(self, config: DictConfig, **kwargs) -> None:
        """
        Init MetricModule as subclass of MetricCollection
        Can directly be initialized with the config

        Parameters
        ----------
        config: DictConfig
        kwargs :
            arguments passed to super class (MetricCollection)
        """

        metrics = {}
        for name, m_conf in config.items():
            if m_conf is not None:
                metric = hydra.utils.instantiate(m_conf)
                metrics[name] = metric
        super().__init__(metrics, compute_groups=False, **kwargs)

    def compute(self):
        dic = super().compute()
        # print(dic)
        wF1 = dic["wF1_Subset1"] * 0.6 + dic["wF1_Subset2"] * 0.2 + dic["wF1_Subset3"] * 0.2
        dic["wF1"] = wF1
        return dic


class ConfusionMatrix(Metric):
    full_state_update = False

    def __init__(self, num_classes: int) -> None:
        """
        Create an empty confusion matrix

        Parameters
        ----------
        num_classes : int
            number of classes inside the Dataset
        """
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.add_state(
            "mat",
            default=torch.zeros((num_classes, num_classes), dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        """
        updating the confmat(self.mat)

        Parameters
        ----------
        pred : torch.Tensor
            prediction (softmax)
        gt : torch.Tensor
            gt mask
        """
        gt = gt.flatten()  # .detach()#.cpu()
        # if softmax input
        pred = pred.argmax(1).flatten()  # .detach()#.cpu()
        # if argmax input
        # pred = pred.flatten()  # .detach()#.cpu()
        n = self.num_classes

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            # Using the torchmetrics implementation of bincount, since the torch one does not
            # support deterministic behaviour
            # confmat = torch.bincount(inds, minlength=n**2).reshape(n, n)
            confmat = _bincount(inds, minlength=n**2).reshape(n, n)
        self.mat += confmat  # .to(self.mat)

    def save_state(self, trainer: pl.Trainer) -> None:
        """
        save the raw and normalized confusion matrix (self.mat) as image/figure to tensorboard
        Adopted from: https://www.tensorflow.org/tensorboard/image_summaries

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer itself to access the logger and parameters like current epoch etc
        """

        def mat_to_figure(mat: np.ndarray, name: str = "Confusion matrix") -> Figure:
            """

            Parameters
            ----------
            mat: np.ndarray
                Confusion Matrix as np array of shape n x n, which n = number of classes
            name: str, optional
                titel of the image

            Returns
            -------
            Figure:
                Visualization of the Confusion matrix
            """
            figure = plt.figure(figsize=(8, 8))
            plt.imshow(mat, interpolation="nearest", cmap=plt.cm.viridis)

            plt.title(name)
            plt.colorbar()
            if hasattr(self, "class_names"):
                labels = self.class_names
            else:
                labels = np.arange(self.num_classes)

            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=90)  # , rotation=45)
            plt.yticks(tick_marks, labels)

            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.close(figure)

            return figure

        # Logging Confusion Matrix
        confmat = self.mat.detach().cpu().numpy()
        figure = mat_to_figure(confmat, "Confusion Matrix")

        trainer.logger.experiment.add_figure(
            "ConfusionMatrix/ConfusionMatrix", figure, trainer.current_epoch
        )

        # Logging the Normalized Confusion Matrix
        confmat_norm = np.around(
            confmat.astype("float") / confmat.sum(axis=1)[:, np.newaxis], decimals=2
        )
        figure = mat_to_figure(confmat_norm, "Confusion Matrix (normalized by row)")
        trainer.logger.experiment.add_figure(
            "ConfusionMatrix/ConfusionMatrix_normalized", figure, trainer.current_epoch
        )


class IoU(ConfusionMatrix):
    def __init__(
        self, per_class: bool = False, labels: list = None, ignore_class: int = None, **kwargs
    ):
        """
        Init the IoU Class as subclass of ConfusionMatrix

        Parameters
        ----------
        per_class : bool, optional
            If False the mean IoU over the classes is returned
            If True additionally the IoU for each class is returned
        labels : list of str, optional
            names of the labels in the dataset
        ignore_class : int, optional
            index of the class which should be ignored during computing the mean IoU
        kwargs:
            arguments passed to the super class (ConfusionMatrix)
        """

        super().__init__(**kwargs)

        self.per_class = per_class
        self.ignore_class = ignore_class
        if labels is not None:
            self.class_names = labels
        if self.per_class:
            if labels is None:
                labels = np.arange(self.num_classes).astype(str)
            if ignore_class is not None:
                labels = np.delete(labels, ignore_class)
            self.labels = ["IoU_" + label for label in labels]

    def get_iou_from_mat(self, confmat: torch.Tensor) -> torch.Tensor:
        """
        Computing the IoU from a confusion matrix (class wise)
        Inset 0 if the IoU of a class is nan
        Delete corresponding entry if there is an ignore_class

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

        if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
            IoU = torch.cat((IoU[: self.ignore_class], IoU[self.ignore_class + 1 :]))
        IoU[IoU.isnan()] = 0.0
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
        mIoU = IoU.mean()
        if self.per_class:
            IoU = {self.labels[i]: IoU[i] for i in range(len(IoU))}
            IoU["mIoU"] = mIoU
            return IoU
        else:
            return mIoU


class Dice_AGGC2022(ConfusionMatrix):
    """
    Metric for AGGC2022 Challenge
    Subclass of ConfusionMatrix
    Computing the weighted F1/Dice Score over the 5 Classes of interest (no score for backroung)
    Adapted from: https://github.com/jydada/AGGC2022/blob/main/eval.py
    """

    def compute(self):
        conf_mat = self.mat
        Stroma_Recall = conf_mat[1, 1] / torch.sum(conf_mat[1, :])
        Normal_Recall = conf_mat[2, 2] / torch.sum(conf_mat[2, :])
        G3_Recall = conf_mat[3, 3] / torch.sum(conf_mat[3, :])
        G4_Recall = conf_mat[4, 4] / torch.sum(conf_mat[4, :])
        G5_Recall = conf_mat[5, 5] / torch.sum(conf_mat[5, :])

        Stroma_Pre = conf_mat[1, 1] / (torch.sum(conf_mat[:, 1]) - conf_mat[0, 1])
        Normal_Pre = conf_mat[2, 2] / (torch.sum(conf_mat[:, 2]) - conf_mat[0, 2])
        G3_Pre = conf_mat[3, 3] / (torch.sum(conf_mat[:, 3]) - conf_mat[0, 3])
        G4_Pre = conf_mat[4, 4] / (torch.sum(conf_mat[:, 4]) - conf_mat[0, 4])
        G5_Pre = conf_mat[5, 5] / (torch.sum(conf_mat[:, 5]) - conf_mat[0, 5])

        F1_Stroma = 2 * Stroma_Pre * Stroma_Recall / (Stroma_Pre + Stroma_Recall)
        F1_Normal = 2 * Normal_Pre * Normal_Recall / (Normal_Pre + Normal_Recall)
        F1_G3 = 2 * G3_Pre * G3_Recall / (G3_Pre + G3_Recall)
        F1_G4 = 2 * G4_Pre * G4_Recall / (G4_Pre + G4_Recall)
        F1_G5 = 2 * G5_Pre * G5_Recall / (G5_Pre + G5_Recall)

        if F1_Stroma.isnan():
            F1_Stroma = 0.0
        if F1_Normal.isnan():
            F1_Normal = 0.0
        if F1_G3.isnan():
            F1_G3 = 0.0
        if F1_G4.isnan():
            F1_G4 = 0.0
        if F1_G5.isnan():
            F1_G5 = 0.0

        Weighted_average_F1score = (
            0.25 * F1_G3 + 0.25 * F1_G4 + 0.25 * F1_G5 + 0.125 * F1_Normal + 0.125 * F1_Stroma
        )
        return {
            "wF1": Weighted_average_F1score,
            "F1_Stroma": F1_Stroma,
            "F1_Normal": F1_Normal,
            "F1_G3": F1_G3,
            "F1_G4": F1_G4,
            "F1_G5": F1_G5,
        }


class Dice_AGGC2022_subsets(Dice_AGGC2022):
    def __init__(self, subset, **kwargs):
        self.subset = subset
        super().__init__(**kwargs)

    def update(self, pred: torch.Tensor, gt: torch.Tensor, subsets: str) -> None:
        # print(subset, self.subset, subset == self.subset)
        # print(torch.where(subset == self.subset, True, False))
        if self.subset in subsets:
            if np.all(subsets == self.subset):
                super().update(pred, gt)
            else:
                indices = np.where(subsets == self.subset)[0]
                # indices = indices[1:]
                # indices = torch.where(subset == self.subset)
                # print(indices)
                # print(indices, pred.shape, gt.shape)
                # if indices != []:
                indices = torch.tensor(indices, device=pred.device)
                pred = torch.index_select(pred, 0, indices)
                gt = torch.index_select(gt, 0, indices)

                # print(pred.shape, gt.shape)
                # if pred != []:
                super().update(pred, gt)

    def compute(self):
        # print(self.subset, self.mat)
        out = super(Dice_AGGC2022_subsets, self).compute()
        out = {k + "_" + self.subset: v for k, v in out.items()}
        return out


class binary_Dice(Metric):
    def __init__(self):
        """
        Binary Dice for the EndoCV2022 challenge with ignoring background (class 0)

        update:
        if gt = 0 and pred = 0, irgnore sample
        if gt = 0 amd pred !=0, return 0
        else compute dice score

        compute:
        mean over the results of each sample
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_classes = 2
        self.add_state("per_image", default=[], dist_reduce_fx="cat")

    def update(self, pred, gt):
        gt = gt.flatten()  # .detach().cpu()
        pred = pred.argmax(1).flatten()  # .detach().cpu()

        with torch.no_grad():
            gt_s = gt.sum()
            pred_s = pred.sum()
            if gt_s == 0 and pred_s == 0:
                dice = None
            elif gt_s == 0 and pred_s != 0:
                dice = torch.tensor(0, device=pred.device)
                self.per_image.append(dice)
            else:
                # print(gt.shape,pred.shape)
                dice = (2 * (gt * pred).sum() + 1e-15) / (gt_s + pred_s + 1e-15)
                self.per_image.append(dice)

    def compute(self):

        self.per_image = dim_zero_cat(self.per_image)

        # log.info("Not None samples %s", len(self.per_image))
        mDice = self.per_image.mean()

        return mDice


class Dice(F1Score):
    # Not tested, use on own risk
    def __init__(
        self,
        num_classes,
        per_class=False,
        labels=None,
        mdmc_average="global",
        multiclass=True,
        ignore_index=None,
        **kwargs
    ):
        # self.per_class=per_class
        # if self.per_class:
        self.per_class = per_class
        self.ignore_class = ignore_index

        if self.per_class:
            kwargs["average"] = None
        super().__init__(
            num_classes=num_classes,
            mdmc_average=mdmc_average,
            multiclass=multiclass,
            ignore_index=ignore_index,
            **kwargs
        )

        if self.per_class:
            if labels is None:
                labels = np.arange(self.num_classes).astype(str)
            if self.ignore_class is not None:
                labels = np.delete(labels, self.ignore_class)
            self.labels = ["Dice_" + label for label in labels]

    def update(self, preds, target):
        target = target.flatten()  # .detach()  # .cpu()
        preds = preds.argmax(1).flatten()  # .detach()  # .cpu()
        k = (target >= 0) & (target < self.num_classes)
        preds = preds[k]
        target = target[k]
        return super().update(preds, target)

    def compute(self):
        Dice = super().compute()
        if self.per_class:
            Dice[Dice.isnan()] = 0.0
            mDice = Dice.mean()
            Dice = {self.labels[i]: Dice[i] for i in range(len(Dice))}
            Dice["mDice"] = mDice
            return Dice
        else:
            return Dice


'''
class Dice_Confmat(ConfusionMatrix):
    # Experimental, use on own risk
    def __init__(
        self, per_class: bool = False, labels: list = None, ignore_class: int = None, **kwargs
    ):
        """
        Init Dice as a sub class of ConfusionMatrix

        Parameters
        ----------
        per_class : bool, optional
            If False the mean IoU over the classes is returned
            If True additionally the IoU for each class is returned
        labels : list of str, optional
            names of the labels in the dataset
        ignore_class : int, optional
            index of the class which should be ignored during computing the mean IoU
        kwargs:
            arguments passed to the super class (ConfusionMatrix)
        """
        super().__init__(**kwargs)
        self.per_class = per_class
        self.ignore_class = ignore_class

        if self.per_class:
            if labels is None:
                labels = np.arange(self.num_classes).astype(str)
            if ignore_class is not None:
                labels = np.delete(labels, ignore_class)
            self.labels = ["Dice_" + label for label in labels]

    def get_dice_from_mat(self, confmat: torch.Tensor) -> torch.Tensor:
        """
        Computing the Dice from a confusion matrix (class wise)
        Inset 0 if the IoU of a class is nan
        Delete corresponding entry if there is an ignore_class

        Parameters
        ----------
        confmat : torch.Tensor

        Returns
        -------
        torch.Tensor :
            Tensor contains the Dice for each class
        """
        if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
            confmat[self.ignore_class] = 0.0
        intersection = confmat.diag()
        sum = confmat.sum(1) + confmat.sum(0)
        dice = 2 * intersection / sum

        if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
            dice = torch.cat((dice[: self.ignore_class], dice[self.ignore_class + 1 :]))

        dice[dice.isnan()] = 0
        return dice

    def compute(self) -> dict or torch.Tensor:
        """
        Compute the Dice from the confusion matrix.
        Depended on initialization return mean Dice and IoU per class or only mean IoU.

        Returns
        -------
        dict or torch.Tensor :
            a single Tensor if only mean IoU is returned, a dict if additionally the class wise
            IoU is returned
        """
        dice = self.get_dice_from_mat(self.mat.clone())
        mdice = dice.mean()
        if self.per_class:
            # print(dice)
            dice = {self.labels[i]: dice[i] for i in range(len(dice))}
            dice["mDice"] = mdice
            return dice
        else:
            return mdice
'''
