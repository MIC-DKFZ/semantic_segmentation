from typing import Dict

import torch
from torchmetrics import MetricCollection

from src.trainer.semantic_segmentation_multilabel_trainer import SegMLModel
from src.utils.utils import get_logger
from src.visualization.utils import show_mask_multilabel_seg

log = get_logger(__name__)


class SegMLModelPL(SegMLModel):
    def update_metric(
        self,
        y_pred: torch.Tensor,
        y_gt: Dict[str, torch.Tensor],
        metric: MetricCollection,
        prefix: str = "",
    ) -> None:
        """
        Update/Call the metric dependent on the metric_cfg.metric_global setting
        If metric_per_img - Iterate through the batch and compute and log metric for each img
        Elif metric_global - Update metric for the current batch

        Parameters
        ----------
        y_pred: torch.Tensor
        y_gt: Dict[str, torch.Tensor]
        metric: MetricCollection
        prefix: str, optional
        """
        if self.metric_cfg.metric_per_img:
            # If metric should be called per img, iterate through the batch to compute and log the
            # metric for each img separately
            for yi_pred, yi_gt, yi_label in zip(y_pred, y_gt["mask"], y_gt["labeled"]):
                metric_step = metric(yi_pred.unsqueeze(0), yi_gt.unsqueeze(0), yi_label)
                # exclude nan since pl uses torch.mean for reduction, this way torch.nanmean is simulated
                metric_step = {k: v for k, v in metric_step.items() if not torch.isnan(v)}
                self.log_dict_epoch(
                    metric_step,
                    prefix=prefix,
                    postfix="_per_img",
                    on_step=False,
                    on_epoch=True,
                )
        elif self.metric_cfg.metric_global:
            # Just update the metric
            metric.update(y_pred, y_gt["mask"], y_gt["labeled"])

    def get_loss(
        self, y_pred: Dict[str, torch.Tensor], y_gt: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Loss of each Output and Lossfunction pair (defined by order in Output dict and
        loss_function list), weight them afterward by the corresponding loss_weight and sum the up
        During Validation only use CE loss for runtime reduction

        Parameters
        ----------
        y_pred : Dict[str, torch.Tensor]
            Output prediction of the network as a dict, where the order inside the dict has to
            match the order of the lossfunction defined in the config
            Shape of each tensor: [batch_size, num_classes, w, h]
        y_gt : Dict[str, torch.Tensor]
            The ground truth segmentation mask
            with shape: [batch_size, w, h]

        Returns
        -------
        torch.Tenor
            weighted sum of the losses of the individual model outputs
        """

        loss = torch.tensor(0.0, device=y_gt["mask"].device)
        if self.training:
            for y, lf, weight in zip(y_pred.values(), self.loss_functions, self.loss_weights):
                loss += lf(y, y_gt["mask"].to(torch.float16), y_gt["labeled"]) * weight
        else:
            for y, lf, weight in zip(y_pred.values(), self.val_loss_functions, self.loss_weights):
                loss += lf(y, y_gt["mask"].to(torch.float16), y_gt["labeled"]) * weight
        return loss

    def viz_data(
        self,
        img: torch.Tensor,
        pred: torch.Tensor,
        gt: torch.Tensor,
        cmap: torch.Tensor,
        output_type: str,
    ) -> torch.Tensor:
        """
        Visualize the Data for logging
        In this Case Prediction and GT are visualized and appended

        Parameters
        ----------
        img: torch.Tensor
        pred: torch.Tensor
        gt: torch.Tensor
        cmap: torch.Tensor
        output_type: str

        Returns
        -------
        torch.Tensor

        """
        threshold = 0.5
        pred = (pred >= threshold).float().detach().cpu()
        gt = gt.cpu()

        gt = show_mask_multilabel_seg(gt, cmap, output_type)
        pred = show_mask_multilabel_seg(pred, cmap, output_type)

        axis = 0 if gt.shape[1] > 2 * gt.shape[0] else 1
        fig = torch.cat((pred, gt), axis)
        return fig
