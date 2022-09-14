import numpy as np
import hydra
from omegaconf import DictConfig
import torch
from torchmetrics import Metric, MetricCollection
import pytorch_lightning as pl
from torchmetrics.utilities.data import _bincount
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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


class ConfusionMatrix(Metric):
    full_state_update = False

    def __init__(self, num_classes: int, labels: list = None) -> None:
        """
        Create an empty confusion matrix

        Parameters
        ----------
        num_classes : int
            number of classes inside the Dataset
        labels : list of str, optional
            names of the labels in the dataset
        """
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes

        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.arange(self.num_classes).astype(str)

        self.add_state(
            "mat",
            default=torch.zeros((num_classes, num_classes), dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        """
        updating the Confusion Matrix(self.mat)

        Parameters
        ----------
        pred : torch.Tensor
            prediction (softmax), with shape [batch_size, num_classes, height, width]
        gt : torch.Tensor
            gt mask, with shape [batch_size, height, width]
        """
        # if softmax input
        pred = pred.argmax(1).flatten()  # .detach()#.cpu()
        # if argmax input
        # pred = pred.flatten()  # .detach()#.cpu()

        gt = gt.flatten()  # .detach()#.cpu()
        n = self.num_classes

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            # Using the torchmetrics implementation of bincount, since the torch one does not support deterministic behaviour
            confmat = _bincount(inds, minlength=n**2).reshape(
                n, n
            )  # torch.bincount(inds, minlength=n**2).reshape(n, n)
        self.mat += confmat  # .to(self.mat)

    def save_state(self, trainer: pl.Trainer) -> None:
        """
        save the raw and normalized confusion matrix (self.mat) as image/figure to tensorboard
        Adopted from: https://www.tensorflow.org/tensorboard/image_summaries

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer itself to access the logger and parameters like current epoch etc.
        """

        def mat_to_figure(mat: np.ndarray, name: str = "Confusion matrix") -> Figure:
            """

            Parameters
            ----------
            mat: np.ndarray
                Confusion Matrix as np array of shape n x n, with n = number of classes
            name: str, optional
                title of the image

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
            plt.xticks(tick_marks, labels, rotation=-90)  # , rotation=45)
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
        figure = mat_to_figure(confmat_norm, "Confusion Matrix (normalized)")
        trainer.logger.experiment.add_figure(
            "ConfusionMatrix/ConfusionMatrix_normalized", figure, trainer.current_epoch
        )


class IoU(ConfusionMatrix):
    def __init__(
        self, per_class: bool = False, name: str = "IoU", replace_nan: bool = True, **kwargs
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
        if self.replace_nan:
            IoU[IoU.isnan()] = 0.0
        mIoU = IoU.mean()

        if self.per_class:
            IoU = {self.name + "_" + self.labels[i]: IoU[i] for i in range(len(IoU))}
            IoU["mean" + self.name] = mIoU
            return IoU
        else:
            return mIoU


class Dice(ConfusionMatrix):
    def __init__(
        self, per_class: bool = False, name: str = "Dice", replace_nan: bool = True, **kwargs
    ):
        """
        Init the Dice Class as subclass of ConfusionMatrix
        Behaviour similar to torchmetric.F1Score(num_classes=6,average="macro",mdmc_average="global",multiclass=True)

        Parameters
        ----------
        per_class : bool, optional
            If False the mean Dice over the classes is returned
            If True additionally the Dice for each class is returned
        name : str, optional
            Name of the metric, used as prefix for logging the class scores
        replace_nan : bool, optional
            replace NaN by 0.0
        kwargs :
            arguments passed to the super class (ConfusionMatrix)
        """

        super().__init__(**kwargs)

        self.per_class = per_class
        self.name = name
        self.replace_nan = replace_nan

    def get_dice_from_mat(self, confmat: torch.Tensor) -> torch.Tensor:
        """
        Computing the Dice from a confusion matrix (class wise)

        Parameters
        ----------
        confmat : torch.Tensor

        Returns
        -------
        torch.Tensor :
            Tensor contains the IoU for each class
        """
        intersection = confmat.diag()
        Dice = 2 * intersection / (confmat.sum(1) + confmat.sum(0))

        # for using a ignore class
        # if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
        #    Dice = torch.cat((Dice[: self.ignore_class], Dice[self.ignore_class + 1 :]))

        return Dice

    def compute(self) -> dict or torch.Tensor:
        """
        Compute the Dice from the confusion matrix.
        Depended on initialization return mean Dice with or without Dice per class
        For Computing the mean set all NaN to 0

        Returns
        -------
        dict or torch.Tensor :
            a single Tensor if only mean IoU is returned, a dict if additionally the class wise
            Dice is returned
        """
        Dice = self.get_dice_from_mat(self.mat.clone())

        if self.replace_nan:
            Dice[Dice.isnan()] = 0.0
        mDice = Dice.mean()

        if self.per_class:
            Dice = {self.name + "_" + self.labels[i]: Dice[i] for i in range(len(Dice))}
            Dice["mean" + self.name] = mDice
            return Dice
        else:
            return mDice
