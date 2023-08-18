import numpy as np
import torch
from torchmetrics import Metric
import lightning as L
from torchmetrics.utilities.data import _bincount
import matplotlib.pyplot as plt

plt.switch_backend("agg")
from matplotlib.figure import Figure


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
        # pred = pred.argmax(1).flatten()  # .detach()#.cpu()
        pred = pred.argmax(1)

        # if argmax input
        # pred = pred.flatten()  # .detach()#.cpu()
        pred = pred.flatten()
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

    def save_state(self, trainer: L.Trainer) -> None:
        """
        save the raw and normalized confusion matrix (self.mat) as image/figure to tensorboard
        Adopted from: https://www.tensorflow.org/tensorboard/image_summaries

        Parameters
        ----------
        trainer : pl.Trainer
            The trainers itself to access the logger and parameters like current epoch etc.
        """

        def mat_to_figure(
            mat: np.ndarray, name: str = "Confusion matrix", normalized: bool = False
        ) -> Figure:
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
            if normalized:
                plt.clim(0, 1)
            plt.colorbar()
            # if hasattr(self, "labels"):
            #     labels = self.labels
            # else:
            #     labels = np.arange(self.num_classes)

            tick_marks = np.arange(len(self.labels))
            plt.xticks(tick_marks, self.labels, rotation=-45)
            plt.yticks(tick_marks, self.labels)

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
        figure = mat_to_figure(confmat_norm, "Confusion Matrix (normalized)", True)
        trainer.logger.experiment.add_figure(
            "ConfusionMatrix/ConfusionMatrix_normalized", figure, trainer.current_epoch
        )
