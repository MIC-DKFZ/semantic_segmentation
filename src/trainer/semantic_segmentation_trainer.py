from typing import Tuple, Dict, List

import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from patchly import GridSampler, Aggregator

from src.utils.utils import get_logger
from src.utils.config_utils import first_from_dict
from src.visualization.utils import show_mask_sem_seg
from src.trainer.base_trainer import BaseModel


log = get_logger(__name__)


class SegModel(BaseModel):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward the input to the model
        If model prediction is not a dict covert the output into a dict

        Parameters
        ----------
        x : torch.Tensor
            Input to predict

        Returns
        -------
        Dict[str, torch.Tensor] :
            Prediction of the model which a separate key for each model output
        """
        x = self.model(x)
        # Covert output to dict if output is list, tuple or tensor
        if not isinstance(x, dict):
            if isinstance(x, list) or isinstance(x, tuple):
                keys = ["out" + str(i) for i in range(len(x))]
                x = dict(zip(keys, x))
            elif isinstance(x, torch.Tensor):
                x = {"out": x}
        return x

    def forwad_tta(
        self,
        x: torch.Tensor,
        patchwise: bool = False,
        patch_size: Tuple[int, int] = None,
        resize: Tuple[int, int] = None,
        scales: List[float] = [1.0],
        h_flip: bool = None,
        v_flip: bool = None,
        overlap: float = 0.5,
        weights: str = "gaussian",
    ):
        if patchwise:
            sampler = GridSampler(
                image=x,
                spatial_size=x.shape[-2:],
                patch_size=patch_size,
                step_size=(
                    int(patch_size[0] * (1 - overlap)),
                    int(patch_size[1] * (1 - overlap)),
                ),
                spatial_first=False,
            )

            # Instantiating the Aggregator to aggregate the patches from the GridSampler
            aggregator = Aggregator(
                sampler=sampler,
                output=torch.zeros(
                    [x.shape[0], self.num_classes, x.shape[-2], x.shape[-1]],
                    device=x.device,
                ),
                spatial_first=False,
                device=x.device,
                weights=weights,
            )
            # Iterate through the patches from sampler, predict them and give to aggregator
            for patch, bbox in sampler:
                pred = self._forward_tta_resize_scale_flip(patch, resize, scales, h_flip, v_flip)
                aggregator.append(pred, bbox)

            return {"out": aggregator.get_output(inplace=True)}
        else:
            return {"out": self._forward_tta_resize_scale_flip(x, resize, scales, h_flip, v_flip)}

    def _forward_tta_resize_scale_flip(
        self,
        x: torch.Tensor,
        resize: Tuple[int, int] = None,
        scales: List[float] = [1.0],
        h_flip: bool = False,
        v_flip: bool = False,
    ) -> Dict[str, torch.Tensor]:

        if resize:
            x_resize = x.size(2), x.size(3)
            if x_resize == resize:
                x = F.interpolate(x, tuple(resize), mode="bilinear", align_corners=True)

        pred = self._forward_tta_scale_flip(x, scales, h_flip, v_flip)

        if resize:
            if x_resize == resize:
                pred = F.interpolate(pred, x_resize, mode="bilinear", align_corners=True)
        return pred

    def _forward_tta_scale_flip(
        self,
        x: torch.Tensor,
        scales: List[float] = [1.0],
        h_flip: bool = False,
        v_flip: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Input is scaled by given scales forwarded to the model (forward_tta_flipping, and pass args
        about tta_flipping)
        Prediction is scaled back to original scale and averaged about each version

        Parameters
        ----------
        x : torch.Tensor
            Input to predict
        scales: List[float], optional
            List of scales for tta
        h_flip: bool, optional
            If horizontal flipping should be used in forward_tta_flipping
        v_flip: bool, optional
            If vertical flipping should be used in forward_tta_flipping
        Returns
        -------
        Dict[str, torch.Tensor] :
            Prediction of the model
        """
        x_size = x.size(2), x.size(3)
        total_pred = None

        # Iterate through the scales and sum the predictions up
        for scale in scales:

            # Scale input to the target scale
            if scale == 1:
                x_scaled = x
            else:
                s_size = int(x_size[0] * scale), int(x_size[1] * scale)
                x_scaled = F.interpolate(x, s_size, mode="bilinear", align_corners=True)

            # Prediction of current scaled image
            y_prediction = self._forward_tta_flip(x_scaled, h_flip, v_flip)

            # Scale prediction back to the original scale
            if scale != 1:
                y_prediction = F.interpolate(
                    y_prediction, x_size, mode="bilinear", align_corners=True
                )

            # Summing the predictions up
            if total_pred == None:
                total_pred = y_prediction
            else:
                total_pred += y_prediction

        # Average the prediction over all scales
        total_pred /= len(scales)

        return total_pred

    def _forward_tta_flip(
        self,
        x: torch.Tensor,
        h_flip: bool = False,
        v_flip: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Input is flipped around defined axis and forwarded to the model
        Prediction is flipped back to original orientation and averaged about each version

        Parameters
        ----------
        x : torch.Tensor
            Input to predict
        h_flip: bool, optional
            If horizontal flipping should be used
        v_flip: bool, optional
            If vertical flipping should be used

        Returns
        -------
        Dict[str, torch.Tensor]
            Prediction of the model (only first output)
        """

        y_prediction = first_from_dict(self(x))

        if not h_flip and not v_flip:
            return y_prediction  # {"out": y_prediction}
        flip_dims = []
        if h_flip:
            flip_dims.append((3,))
        if v_flip:
            flip_dims.append((2,))
        if h_flip and v_flip:
            flip_dims.append((2, 3))

        for flip_dim in flip_dims:
            y_prediction += torch.flip(first_from_dict(self(torch.flip(x, flip_dim))), flip_dim)

        y_prediction /= len(flip_dims) + 1
        return y_prediction

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Forward the image through the model and compute the loss
        (Optional) update the metric stepwise of global (defined by metric_call parameter)

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Contains img (shape==[batch_size,num_classes,w,h]) and mask (shape==[batch_size,w,h])
        batch_idx : int
            Index of the batch

        Returns
        -------
        torch.Tensor :
            Training loss
        """
        # Model - Predict the input batch
        x, y_gt = batch
        y_pred = self(x)
        # Loss - Compute and log the train loss
        loss = self.get_loss(y_pred, y_gt)
        self.log(
            name="Loss/training_loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        # Metric - (Optional) update train metric
        if self.metric_cfg.train_metric:
            self.update_metric(
                list(y_pred.values())[0], y_gt, self.metric_train, prefix="metric_train/"
            )

        # Loss - Return
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Forward the image through the model and compute the loss
        Update the metric stepwise of global (defined by metric_call parameter)
        Patchwise inference is used if defined in self.tta_cfg

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Contains img (shape==[batch_size,num_classes,w,h]) and mask (shape==[batch_size,w,h])
        batch_idx : int
            Index of the batch

        Returns
        -------
        torch.Tensor :
            Validation loss
        """
        # Model - Predict the input batch - use patchwise inference if defined self.tta_cfg
        x, y_gt = batch
        y_pred = self.forwad_tta(
            x,
            patchwise=self.tta_cfg.patchwise,
            patch_size=self.tta_cfg.patch_size,
            resize=self.tta_cfg.resize,
            scales=[1],
            h_flip=False,
            v_flip=False,
            overlap=self.tta_cfg.patch_overlap,
            weights="gaussian",
        )

        # Loss - Compute and log the validation loss
        val_loss = self.get_loss(y_pred, y_gt)
        self.log(
            name="Loss/validation_loss",
            value=val_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        # Metric - update validation metric
        self.update_metric(first_from_dict(y_pred), y_gt, self.metric, prefix="metric/")

        # Logging - log example predictions
        if self.global_rank == 0 and not self.trainer.sanity_checking:
            self.log_batch_prediction(x, y_pred, y_gt, batch_idx)

        # Loss - Return
        return val_loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Forward the image through the model and compute the loss
        Update the metric stepwise of global (defined by metric_call parameter)
        Patchwise inference (+test time augmentation) is used if defined in self.tta_cfg

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Contains img (shape==[batch_size,num_classes,w,h]) and mask (shape==[batch_size,w,h])
        batch_idx : int
            Index of the batch

        Returns
        -------
        torch.Tensor :
            Validation loss
        """
        # Model - Predict the input batch - use patchwise inference (+tta) as defined in self.tta_cfg
        x, y_gt = batch
        y_pred = self.forwad_tta(
            x,
            patchwise=self.tta_cfg.patchwise,
            patch_size=self.tta_cfg.patch_size,
            resize=self.tta_cfg.resize,
            scales=self.tta_cfg.scales,
            h_flip=self.tta_cfg.hflip,
            v_flip=self.tta_cfg.vflip,
            overlap=self.tta_cfg.patch_overlap,
            weights="gaussian",
        )

        # Loss - Compute and log the test loss
        test_loss = self.get_loss(y_pred, y_gt)
        self.log(
            name="Loss/Test_loss",
            value=test_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        # Metric - update test metric
        self.update_metric(first_from_dict(y_pred), y_gt, self.metric, prefix="metric_test/")

        # Logging - log example predictions
        if self.global_rank == 0:
            self.log_batch_prediction(x, first_from_dict(y_pred), y_gt, batch_idx)

        # Loss - Return
        return test_loss

    def predict_step(
        self, batch: Tuple[torch.Tensor, str], batch_idx: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
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
        # print(batch.shape)
        img = batch[0]
        y_pred = self.forwad_tta(
            img,
            patchwise=self.tta_cfg.patchwise,
            patch_size=self.tta_cfg.patch_size,
            resize=self.tta_cfg.resize,
            scales=self.tta_cfg.scales,
            h_flip=self.tta_cfg.hflip,
            v_flip=self.tta_cfg.vflip,
            overlap=self.tta_cfg.patch_overlap,
            weights="gaussian",
        )
        return y_pred

    def get_loss(self, y_pred: Dict[str, torch.Tensor], y_gt: torch.Tensor) -> torch.Tensor:
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
        y_gt : torch.Tensor
            The ground truth segmentation mask
            with shape: [batch_size, w, h]

        Returns
        -------
        torch.Tenor
            weighted sum of the losses of the individual model outputs
        """
        loss = torch.tensor(0.0, device=y_gt.device)
        if self.training:
            for y, lf, weight in zip(y_pred.values(), self.loss_functions, self.loss_weights):
                loss += lf(y, y_gt.long()) * weight
        else:
            for y, lf, weight in zip(y_pred.values(), self.val_loss_functions, self.loss_weights):
                loss += lf(y, y_gt.long()) * weight
        return loss

    def update_metric(
        self,
        y_pred: torch.Tensor,
        y_gt: torch.Tensor,
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
        y_gt: torch.Tensor
        metric: MetricCollection
        prefix: str, optional
        """

        # Metric - Update metric dependent on if it should be called per image or globally
        if self.metric_cfg.metric_per_img:

            # Metric - Iterate through the batch and compute and log metric for each img
            for yi_pred, yi_gt in zip(y_pred, y_gt):
                metric_step = metric(yi_pred.unsqueeze(0), yi_gt.unsqueeze(0))
                metric_step = {k: v for k, v in metric_step.items() if not torch.isnan(v)}
                self.log_dict_epoch(
                    metric_step,
                    prefix=prefix,
                    postfix="_per_img",
                    on_step=False,
                    on_epoch=True,
                )
        elif self.metric_cfg.metric_global:
            # Metric - Update metric for the current batch
            metric.update(y_pred, y_gt)
