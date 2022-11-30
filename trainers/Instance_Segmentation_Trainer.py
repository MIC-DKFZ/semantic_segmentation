from omegaconf import DictConfig

import torch
import torch.nn.functional as F

from src.utils import get_logger
from trainers.Semantic_Segmentation_Trainer import SegModel
from src.visualization import show_prediction_inst_seg, show_mask_inst_seg, show_img

log = get_logger(__name__)


class InstModel(SegModel):
    def __init__(self, model_config: DictConfig) -> None:
        """
        __init__ the LightningModule
        instantiate the model and the metric(s)

        Parameters
        ----------
        config : omegaconf.DictConfig
        """
        super().__init__(model_config)
        if self.train_metric:
            log.info(
                "Training Metric for Instance Segmentation is not supported and is set to False"
            )
            self.train_metric = False

    def forward(self, x: torch.Tensor, gt=None) -> dict:
        """
        forward the input to the model

        Parameters
        ----------
        x : torch.Tensor
            input to predict
        gt : dict of {str:torch.Tensor}
            target, only used for training to compute the loss

        Returns
        -------
        if training
            torch.Tensor: training loss
        else:
            dict of {str:torch.Tensor} :
                prediction of the model containing masks, boxes, scores and labels
        """
        if self.training:
            x = self.model(x, gt)
        else:
            x = self.model(x)

        return x

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
        x, y_gt = batch
        loss_dict = self(x, y_gt)

        loss = sum(l for l in loss_dict.values())

        # compute and log loss
        self.log(
            "Loss/training_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        return loss

    def validation_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        """
        Forward the image through the model and compute the loss
        update the metric stepwise of global (defined by metric_call parameter)

        Parameters
        ----------
        batch : list of dicts
        batch_idx : int
            index of the batch

        Returns
        -------
        """

        # predict batch
        x, y_gt = batch
        y_pred = self(x)

        # update validation metric
        self.update_metric(y_pred, y_gt, self.metric, prefix="metric/")

        # log some example predictions to tensorboard
        # ensure that exactly self.num_example_predictions examples are taken
        if self.global_rank == 0 and not self.trainer.sanity_checking:
            self.log_batch_prediction(x, y_pred, y_gt, batch_idx)

    def on_test_start(self) -> None:
        pass

    def test_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        """
        copy of validation step
        """
        # predict batch
        x, y_gt = batch
        y_pred = self(x)

        # update validation metric
        self.update_metric(y_pred, y_gt, self.metric, prefix="metric_test/")

        # log some example predictions to tensorboard
        if self.global_rank == 0 and not self.trainer.sanity_checking:
            self.log_batch_prediction(x, y_pred, y_gt, batch_idx)

    def get_loss(self, y_pred: dict, y_gt: torch.Tensor) -> torch.Tensor:
        pass

    def log_batch_prediction(self, imgs: list, preds: list, gts: list, batch_idx: int) -> None:
        """
        logging example prediction and gt to tensorboard

        Parameters
        ----------
        imgs: [torch.Tensor]
        pred : [dict]
        gt : [dict]
        batch_idx: int
            idx of the current batch, needed for naming of the predictions
        """
        print(type(imgs[0]), type(preds[0]), type(gts[0]))
        # Check if the current batch has to be logged, if yes how many images
        val_batch_size = self.trainer.datamodule.val_batch_size
        diff_to_show = self.num_example_predictions - (batch_idx * val_batch_size)
        if diff_to_show > 0:
            current_batche_size = len(imgs)
            # log the desired number of images
            for i in range(min(current_batche_size, diff_to_show)):
                img = imgs[i].detach().cpu()

                pred = preds[i]
                pred = [{k: v.detach().cpu() for k, v in pred.items()}]

                gt = gts[i]
                gt = [{k: v.detach().cpu() for k, v in gt.items()}]

                # colormap class labels and transform image
                pred = show_prediction_inst_seg(pred, img.shape[-2:], output_type="torch")
                gt = show_mask_inst_seg(gt[0], img.shape[-2:], output_type="torch")
                img = show_img(img, output_type="torch")

                alpha = 0.5
                gt = (img * alpha + gt * (1 - alpha)).type(torch.uint8)
                pred = (img * alpha + pred * (1 - alpha)).type(torch.uint8)

                # concat pred and gt for better visualization
                axis = 0 if gt.shape[1] > 2 * gt.shape[0] else 1
                fig = torch.cat((pred, gt), axis)

                # resize fig for not getting to large tensorboard-files
                w, h, c = fig.shape
                max_size = 2048
                if max(w, h) > max_size:
                    s = max_size / max(w, h)

                    fig = fig.permute(2, 0, 1).unsqueeze(0).float()
                    fig = F.interpolate(fig, size=(int(w * s), int(h * s)), mode="nearest")
                    fig = fig.squeeze(0).permute(1, 2, 0).to(torch.uint8)

                # Log Figure to tensorboard
                self.trainer.logger.experiment.add_image(
                    "Example_Prediction/prediction_gt__sample_"
                    + str(batch_idx * val_batch_size + i),
                    fig,
                    self.current_epoch,
                    dataformats="HWC",
                )
