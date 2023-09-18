from typing import Tuple, Dict, List, Union

from omegaconf import DictConfig
import torch
from torchmetrics import MetricCollection

from src.trainer.base_trainer import BaseModel
from src.utils.utils import get_logger
from src.utils.visualization import show_prediction_inst_seg, show_mask_inst_seg

log = get_logger(__name__)


class InstModel(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Init the LightningModule

        Parameters
        ----------
        cfg : DictConfig
        """
        super().__init__(cfg)
        if self.metric_cfg.train_metric:
            log.info(
                "Training Metric for Instance Segmentation is not supported and is set to False"
            )
            self.metric_cfg.train_metric = False

    def forward(
        self, x: List[torch.Tensor], gt: Union[None, List[Dict[str, torch.Tensor]]] = None
    ) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Forward the input to the model
        During training the loss is returned, else the prediction

        Parameters
        ----------
        x : List[torch.Tensor]
            input to predict
        gt : Union[None, List[Dict[str, torch.Tensor]]
            List[Dict[str, torch.Tensor] During Training the gt is a List of Dicts of Tensor
            None else

        Returns
        -------
        Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
            Dict[str, torch.Tensor] during training the loss is returned
            List[Dict[str, torch.Tensor]: Prediction of the model containing masks, boxes, scores and labels
        """
        if self.training:
            x = self.model(x, gt)
        else:
            x = self.model(x)
        return x

    def training_step(
        self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> torch.Tensor:
        """
        Forward the image through the model and compute the loss

        Parameters
        ----------
        batch : Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]
            contains img and target dict
        batch_idx : int
            index of the batch

        Returns
        -------
        torch.Tensor :
            training loss
        """
        # Predict batch
        x, y_gt = batch
        loss_dict = self(x, y_gt)

        # Sum up the different losses
        # loss = sum(l for l in loss_dict.values())
        loss = torch.tensor(0.0, device=x[0].device)
        for loss_i in loss_dict.values():
            loss += loss_i

        # Compute and log loss
        self.log(
            "Loss/training_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        return loss

    def validation_step(
        self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> None:
        """
        Forward the image through the model and update the metric

        Parameters
        ----------
        batch : Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]
            contains img and target dict
        batch_idx : int
            index of the batch

        Returns
        -------
        """

        # Predict batch
        x, y_gt = batch
        y_pred = self(x)
        # Update validation metric
        self.update_metric(y_pred, y_gt, self.metric, prefix="metric/")

        # Log some example predictions to tensorboard
        # Ensure that exactly self.num_example_predictions examples are taken
        if self.global_rank == 0 and not self.trainer.sanity_checking:
            self.log_batch_prediction(x, y_pred, y_gt, batch_idx)

    def test_step(
        self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> None:
        """
         Forward the image through the model and update the metric

        Parameters
        ----------
        batch : Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]
            contains img and target dict
        batch_idx : int
            index of the batch

        Returns
        -------
        """
        # Predict batch
        x, y_gt = batch
        y_pred = self(x)

        # Update validation metric
        self.update_metric(y_pred, y_gt, self.metric, prefix="metric_test/")

        # Log some example predictions to tensorboard
        if self.global_rank == 0 and not self.trainer.sanity_checking:
            self.log_batch_prediction(x, y_pred, y_gt, batch_idx)

    def predict_step(
        self, batch: Tuple[torch.Tensor, str], batch_idx: int, *args, **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Forward the image through the model
        Update the metric stepwise of global (defined by metric_call parameter)
        Patchwise inference (+tta) is used if defined in self.tta_cfg

        Parameters
        ----------
        batch : Tuple[torch.Tensor, str]
            Contains img (shape==[batch_size,num_classes,w,h]) and name of the img
        batch_idx : int
            Index of the batch

        Returns
        -------
        Tuple[Dict[str : torch.Tensor], str] :
            Prediction of the model and name of the img
        """
        # Model - Predict the input batch - use patchwise inference (+tta) as defined in self.tta_cfg
        img = batch[0]
        pred = self(img)
        return pred

    def update_metric(
        self,
        y_pred: List[Dict[str, torch.Tensor]],
        y_gt: List[Dict[str, torch.Tensor]],
        metric: MetricCollection,
        prefix: str = "",
    ):
        """
        Update/Call the metric dependent on the metric_cfg.metric_global setting
        If metric_per_img - Iterate through the batch and compute and log metric for each img
        Elif metric_global - Update metric for the current batch

        Parameters
        ----------
        y_pred: List[Dict[str, torch.Tensor]]
        y_gt: List[Dict[str, torch.Tensor]]
        metric: MetricCollection
        prefix: str, optional
        """
        for y in y_pred:
            y["masks"] = y["masks"].squeeze(1)
            for i in range(0, len(y["masks"])):
                x = torch.where(y["masks"][i] >= 0.5, 1, 0)
                y["masks"][i] = x
            y["masks"] = y["masks"].type(torch.uint8)

        if self.metric_cfg.metric_per_img:
            # If metric should be called per img, iterate through the batch to compute and log the
            # metric for each img separately
            for yi_pred, yi_gt in zip(y_pred, y_gt):
                metric_step = metric(yi_pred.unsqueeze(0), yi_gt.unsqueeze(0))
                # Exclude nan since pl uses torch.mean for reduction, this way torch.nanmean is simulated
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
            metric.update(y_pred, y_gt)

    def get_loss(self, y_pred: dict, y_gt: torch.Tensor) -> torch.Tensor:
        pass

    def viz_data(
        self,
        img: torch.Tensor,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
        cmap: torch.Tensor,
        output_type: str,
    ) -> torch.Tensor:
        """
        Visualize the Data for logging
        In this Case Prediction and GT are visualized and appended

        Parameters
        ----------
        img: torch.Tensor
        pred: Dict[str, torch.Tensor]
        gt: Dict[str, torch.Tensor]
        cmap: torch.Tensor
        output_type: str

        Returns
        -------
        torch.Tensor

        """
        img = img.detach().cpu()
        pred = {k: v.detach().cpu() for k, v in pred.items()}
        gt = {k: v.detach().cpu() for k, v in gt.items()}

        pred = show_prediction_inst_seg(pred, img.shape[-2:], output_type=output_type)
        gt = show_mask_inst_seg(gt, img.shape[-2:], output_type=output_type)

        axis = 0 if gt.shape[1] > 2 * gt.shape[0] else 1
        fig = torch.cat((pred, gt), axis)
        return fig
