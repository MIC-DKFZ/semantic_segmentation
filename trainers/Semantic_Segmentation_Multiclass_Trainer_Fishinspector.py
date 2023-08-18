from trainers.Semantic_Segmentation_Multiclass_Trainer import SegMCModel

import torch
from src.metric.metric import MetricModule
from src.utils.utils import get_logger
from src.utils.config_utils import first_from_dict

log = get_logger(__name__)


class SegMCModel(SegMCModel):
    def training_step(self, batch: list, batch_idx: int):
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
        x, y_gt, labeled_classes = batch
        y_pred = self(x)
        loss = self.get_loss(y_pred, y_gt, labeled_classes)
        self.log(
            "Loss/training_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        # (optional) update train metric
        if self.train_metric:
            self.update_metric(
                list(y_pred.values())[0], y_gt, self.metric_train, prefix="metric_train/"
            )

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
        x, y_gt, labeled_classes = batch
        y_pred = self(x)
        y_gt = y_gt
        # compute and log loss to tensorboard
        val_loss = self.get_loss(y_pred, y_gt, labeled_classes)
        self.log(
            name="Loss/validation_loss",
            value=val_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        # update validation metric
        self.update_metric(
            first_from_dict(y_pred),
            y_gt.long(),
            labeled_classes,
            self.metric,
            prefix="metric/",
        )

        # log some example predictions to tensorboard
        if self.global_rank == 0 and not self.trainer.sanity_checking:
            self.log_batch_prediction(x, first_from_dict(y_pred), y_gt, batch_idx)

    def update_metric(
        self,
        y_pred: torch.Tensor,
        y_gt: torch.Tensor,
        labeled_classes,
        metric: MetricModule,
        prefix: str = "",
    ):

        if self.metric_call_stepwise:
            # Log the metric result for each step
            metric_step = metric(y_pred, y_gt, labeled_classes)
            # exclude nan since pl uses torch.mean for reduction, this way torch.nanmean is simulated
            metric_step = {k: v for k, v in metric_step.items() if not torch.isnan(v)}
            self.log_dict_epoch(
                metric_step,
                prefix=prefix,
                postfix="_stepwise",
                on_step=False,
                on_epoch=True,
            )
        elif self.metric_call_per_img:
            # If metric should be called per img, iterate through the batch to compute and log the
            # metric for each img separately
            for yi_pred, yi_gt, lc in zip(y_pred, y_gt, labeled_classes):
                metric_step = metric(yi_pred.unsqueeze(0), yi_gt.unsqueeze(0), lc)
                # exclude nan since pl uses torch.mean for reduction, this way torch.nanmean is simulated
                metric_step = {k: v for k, v in metric_step.items() if not torch.isnan(v)}
                self.log_dict_epoch(
                    metric_step,
                    prefix=prefix,
                    postfix="_per_img",
                    on_step=False,
                    on_epoch=True,
                )
        elif self.metric_call_global:
            # Just update the metric
            metric.update(y_pred, y_gt, labeled_classes)

    def get_loss(self, y_pred: dict, y_gt: torch.Tensor, labeled_classes) -> torch.Tensor:
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

        def get_single_loss(y_pred, y_gt, lf, labeled_classes):

            s_loss = lf(y_pred, y_gt.float(), labeled_classes)
            return s_loss

        loss = sum(
            [
                get_single_loss(y, y_gt.long(), self.loss_functions[i], labeled_classes)
                * self.loss_weights[i]
                for i, y in enumerate(y_pred.values())
            ]
        )

        return loss
