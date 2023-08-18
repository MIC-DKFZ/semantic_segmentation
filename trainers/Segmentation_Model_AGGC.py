from trainers.Semantic_Segmentation_Trainer import SegModel
import torch
import torch.nn.functional as F
from src.utils.utils import get_logger
from src.metric.metric import MetricModule_AGGC
import numpy as np

log = get_logger(__name__)


class SegModel_AGGC(SegModel):
    def __init__(self, model_config) -> None:
        """
        __init__ the LightningModule
        instantiate the model and the metric(s)

        Parameters
        ----------
        config : omegaconf.DictConfig
        """
        super().__init__(model_config)
        self.metric = MetricModule_AGGC(self.config.METRIC.METRICS)

    def training_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        """
        Forward the image through the model and compute the loss
        (optional) update the metric stepwise of global (defined by metric_call parameter)

        Parameters
        ----------
        batch : list of torch.Tensor
            contains img (shape==[batch_size,num_classes,w,h]) and mask (shape==[batch_size,w,h])
        batch_idx : int
            index of the batch

        Returns
        -------
        torch.Tensor :
            training loss
        """
        # predict batch
        x, y_gt, subset = batch
        y_pred = self(x)

        # compute and log loss
        if "CE_AGGC" in self.config.lossfunction or "wCE_AGGC" in self.config.lossfunction:
            loss = self.get_loss(y_pred, y_gt, subset)
        else:
            loss = self.get_loss(y_pred, y_gt)
        self.log(
            "Loss/training_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        # (optional) update train metric
        if hasattr(self, "metric_train"):
            if self.metric_call in ["stepwise", "global_and_stepwise"]:
                # update global metric and log stepwise metric
                metric_step = self.metric_train(list(y_pred.values())[0], y_gt)
                self.log_dict_epoch(
                    metric_step,
                    prefix="metric_train/",
                    postfix="_stepwise",
                    on_step=False,
                    on_epoch=True,
                )
            elif self.metric_call in ["global"]:
                # update only global metric
                self.metric_train.update(list(y_pred.values())[0], y_gt)

        return loss

    def validation_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        """
        Forward the image through the model and compute the loss
        update the metric stepwise of global (defined by metric_call parameter)

        Parameters
        ----------
        batch : list of torch.Tensor
            contains img (shape==[batch_size,num_classes,w,h]) and mask (shape==[batch_size,w,h])
        batch_idx : int
            index of the batch

        Returns
        -------
        torch.Tensor :
            validation loss
        """

        # predict batch
        x, y_gt, subset = batch
        subset = np.array(subset)
        # subset = torch.tensor(subset)

        if not torch.any(y_gt):
            self.log(
                name="Loss/validation_loss",
                value=torch.tensor(0.0),
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True if self.trainer.num_devices > 1 else False,
            )
            return
        else:
            y_pred = self(x)

            # compute and log loss to tensorboard
            val_loss = self.get_loss(y_pred, y_gt)
            self.log(
                name="Loss/validation_loss",
                value=val_loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True if self.trainer.num_devices > 1 else False,
            )  # ,prog_bar=True)

            # update validation metric

            if self.metric_call in ["stepwise", "global_and_stepwise"]:
                # update global metric and log stepwise metric to tensorboard
                metric_step = self.metric(list(y_pred.values())[0], y_gt, subset)
                self.log_dict_epoch(
                    metric_step,
                    prefix="metric/",
                    postfix="_stepwise",
                    on_step=False,
                    on_epoch=True,
                )
            elif self.metric_call in ["global"]:
                # update only global metric
                self.metric.update(list(y_pred.values())[0], y_gt, subset)

            # log some example predictions to tensorboard
            # ensure that exactly self.num_example_predictions examples are taken
            batch_size = y_gt.shape[0]
            if (
                (batch_size * batch_idx) < self.num_example_predictions
                and self.global_rank == 0
                and not self.trainer.sanity_checking
            ):
                self.log_batch_prediction(
                    y_pred["out"],
                    y_gt,
                    batch_idx,
                    self.num_example_predictions - (batch_size * batch_idx),
                )

    def test_step(self, batch, batch_idx):
        """
        For each scale used during testing:
            resize the image to the desired scale
            forward it to the model
            resize to original size
            (optional) flip the the input image and repeat the above steps
            summing up the prediction
        update the metric stepwise of global (defined by metric_call parameter)

        Parameters
        ----------
        batch : list of torch.Tensor
            contains img (shape==[batch_size,num_classes,w,h]) and mask (shape==[batch_size,w,h])
        batch_idx : int
            index of the batch

        Returns
        -------
        torch.Tensor :
            validation loss
        """
        x, y_gt, subset = batch
        subset = np.array(subset)
        if not torch.any(y_gt):
            self.log(
                name="Loss/validation_loss",
                value=torch.tensor(0.0),
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True if self.trainer.num_devices > 1 else False,
            )
            return
        else:
            x_size = x.size(2), x.size(3)
            total_pred = None

            # iterate through the scales and sum the predictions up
            for scale in self.test_scales:
                s_size = int(x_size[0] * scale), int(x_size[1] * scale)
                x_s = F.interpolate(
                    x, s_size, mode="bilinear", align_corners=True
                )  # self.config.MODEL.ALIGN_CORNERS)
                y_pred = self(x_s)  # ["out"]
                y_pred = list(y_pred.values())[0]
                y_pred = F.interpolate(
                    y_pred, x_size, mode="bilinear", align_corners=True
                )  # =self.config.MODEL.ALIGN_CORNERS)

                # if flipping is used the average over the predictions from the flipped and not flipped image is taken
                if self.test_flip:
                    x_flip = torch.flip(x_s, [3])  #

                    y_flip = self(x_flip)  # ["out"]
                    y_pred = list(y_pred.values())[0]
                    y_flip = torch.flip(y_flip, [3])
                    y_flip = F.interpolate(y_flip, x_size, mode="bilinear", align_corners=True)
                    # align_corners=self.config.MODEL.ALIGN_CORNERS)
                    y_pred += y_flip
                    y_pred /= 2

                # summing the predictions up
                if total_pred == None:
                    total_pred = y_pred  # .detach()
                else:
                    total_pred += y_pred  # .detach()

            # update the metric with the aggregated prediction
            if self.metric_call in ["stepwise", "global_and_stepwise"]:
                # update global metric and log stepwise metric to tensorboard
                metric_step = self.metric(total_pred, y_gt, subset)
                self.log_dict_epoch(
                    metric_step,
                    prefix="metric_test/",
                    postfix="_stepwise",
                    on_step=False,
                    on_epoch=True,
                )
            elif self.metric_call in ["global"]:
                # update only global metric
                self.metric.update(total_pred, y_gt, subset)

            # compute and return loss of final prediction
            test_loss = F.cross_entropy(
                total_pred, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX
            )
            self.log(
                "Loss/Test_loss",
                test_loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True if self.trainer.num_devices > 1 else False,
            )

            # log some example predictions to tensorboard
            # ensure that exactly self.num_example_predictions examples are taken
            batch_size = y_gt.shape[0]
            if (batch_size * batch_idx) < self.num_example_predictions and self.global_rank == 0:
                self.log_batch_prediction(
                    total_pred,
                    y_gt,
                    batch_idx,
                    self.num_example_predictions - (batch_size * batch_idx),
                )

            return test_loss

    def get_loss(self, y_pred: dict, y_gt: torch.Tensor, subset=None) -> torch.Tensor:
        """
        Compute Loss of each Output and Lossfunction pair (defined by order in Output dict and
        loss_function list), weight them afterward by the corresponding loss_weight and sum the up
        During Validation only use CE loss for runtime reduction

        Parameters
        ----------
        y_pred : dict of {str: torch.Tensor}
            Output prediction of the network as a dict, where the order inside the dict has to
            match the order of the lossfunction defined in the config
            Shape of each tensor: [batch_size, num_classes, w, h]
        y_gt : torch.Tensor
            The ground truth segmentation mask
            with shape: [batch_size, w, h]

        Returns
        -------
        torch.Tenor
            weighted sum of the losses of the individual model outputs
        """
        if subset is None:
            if self.training:
                loss = sum(
                    [
                        self.loss_functions[i](y, y_gt) * self.loss_weights[i]
                        for i, y in enumerate(y_pred.values())
                    ]
                )
            else:
                loss = sum(
                    [
                        F.cross_entropy(y, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX)
                        * self.loss_weights[i]
                        for i, y in enumerate(y_pred.values())
                    ]
                )
        else:
            if self.training:
                loss = sum(
                    [
                        self.loss_functions[i](y, y_gt, subset) * self.loss_weights[i]
                        for i, y in enumerate(y_pred.values())
                    ]
                )
            else:
                loss = sum(
                    [
                        F.cross_entropy(y, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX)
                        * self.loss_weights[i]
                        for i, y in enumerate(y_pred.values())
                    ]
                )
        return loss
