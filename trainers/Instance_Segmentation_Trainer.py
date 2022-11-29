import hydra
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from matplotlib import cm

from src.metric import MetricModule
from src.loss_function import get_loss_function_from_cfg
from src.utils import has_not_empty_attr, has_true_attr
from src.utils import get_logger
from trainers.Semantic_Segmentation_Trainer import SegModel
from src.visualization import show_prediction_inst_seg, show_mask_inst_seg, show_img
import numpy as np
import cv2

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
        x, y_gt = batch
        y_pred = self(x)

        # update validation metric
        self.update_metric(y_pred, y_gt, self.metric, prefix="metric/")

        # log some example predictions to tensorboard
        # ensure that exactly self.num_example_predictions examples are taken
        # print(len(x))
        batch_size = len(x)
        if (
            (batch_size * batch_idx) < self.num_example_predictions
            and self.global_rank == 0
            and not self.trainer.sanity_checking
        ):
            self.log_batch_prediction(
                x,
                y_pred,
                y_gt,
                batch_idx,
                self.num_example_predictions - (batch_size * batch_idx),
            )

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
        # ensure that exactly self.num_example_predictions examples are taken
        # print(len(x))
        batch_size = len(x)
        if (
            (batch_size * batch_idx) < self.num_example_predictions
            and self.global_rank == 0
            and not self.trainer.sanity_checking
        ):
            self.log_batch_prediction(
                x,
                y_pred,
                y_gt,
                batch_idx,
                self.num_example_predictions - (batch_size * batch_idx),
            )

    def get_loss(self, y_pred: dict, y_gt: torch.Tensor) -> torch.Tensor:
        pass

    def log_batch_prediction(
        self,
        imgs: torch.Tensor,
        preds: torch.Tensor,
        gts: torch.Tensor,
        batch_idx: int = 0,
        max_number: int = 5,
    ) -> None:
        """
        logging example prediction and gt to tensorboard

        Parameters
        ----------
        pred : torch.Tensor
        gt : torch.Tensor
        batch_idx: int, optional
            idx of the current batch, needed for naming of the predictions
        max_number : int, optional
            number of example predictions
        """

        def show_data(img, target, alpha=0.5):
            masks = target["masks"].detach().cpu().squeeze(1)
            boxes = target["boxes"].detach().cpu()
            img = np.array(img.detach().cpu()) * 255
            img = img.transpose((1, 2, 0)).astype(np.uint8)
            for mask, box in zip(masks, boxes):
                color = np.random.randint(0, 255, 3)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(
                    img, (x1, y1), (x2, y2), [int(color[0]), int(color[1]), int(color[2])]
                )
                # cont,_=cv2.findContours(np.array(mask),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(img, cont, 0, [int(color[0]), int(color[1]), int(color[2])], 1)
                x, y = np.where(np.array(mask) == 1)
                img[x, y] = img[x, y] * alpha + color * (1 - alpha)
            return img

        def show_prediction(img, pred, alpha=0.5):
            img = np.array(img.detach().cpu()) * 255  # [0]
            img = img.transpose((1, 2, 0)).astype(np.uint8)

            masks = pred["masks"].detach().cpu().squeeze(1)
            boxes = pred["boxes"].detach().cpu()
            scores = pred["scores"].detach().cpu()

            masks = [mask for mask, score in zip(masks, scores) if score >= 0.5]
            boxes = [box for box, score in zip(boxes, scores) if score >= 0.5]

            for mask, box in zip(masks, boxes):
                # mask = np.array(mask.detach().cpu())[0]
                color = np.random.randint(0, 255, 3)

                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                # cv2.rectangle(img, (x1, y1), (x2, y2), [int(color[0]), int(color[1]), int(color[2])])

                x, y = np.where(mask >= 0.5)
                img[x, y] = img[x, y] * alpha + color * (1 - alpha)

            return img

        batch_size = len(imgs)
        for i in range(min(batch_size, max_number)):
            img = imgs[i].detach().cpu()

            pred = preds[i]
            pred = [{k: v.detach().cpu() for k, v in pred.items()}]

            gt = gts[i]
            gt = [{k: v.detach().cpu() for k, v in gt.items()}]

            img_shape = img.shape[-2:]

            p = show_prediction_inst_seg(pred, img_shape, output_type="torch")
            g = show_mask_inst_seg(gt[0], img_shape, output_type="torch")
            im = show_img(img, output_type="torch")

            p = (im * 0.5 + p * 0.5).type(torch.uint8)
            g = (im * 0.5 + g * 0.5).type(torch.uint8)

            # p = torch.tensor(p.astype(np.uint8))
            # g = torch.tensor(g.astype(np.uint8))
            # print(p.dtype, torch.min(p), torch.max(p))
            # p = pred[i]
            # g = gt[i]
            # im = img[i]
            # g = show_data(im, g)  # gt.detach().cpu()
            # p = show_prediction(im, p)  # pred.argmax(1).detach().cpu()
            # p = torch.tensor(p)
            # g = torch.tensor(g)
            # max_size = 1024
            # w, h, c = p.shape
            # if max(w, h) > max_size:
            #     s = max_size / max(w, h)
            #     w_s, h_s = int(w * s), int(h * s)
            #     print(p.shape, g.shape)
            #     # print(p.unsqueeze(0).shape, g.unsqueeze(0).shape)
            #     g = (
            #         F.interpolate(g.permute(2, 0, 1).unsqueeze(0), size=(w_s, h_s), mode="nearest")
            #         .long()
            #         .squeeze(0)
            #         .permute(1, 2, 0)
            #     )
            #     p = (
            #         F.interpolate(p.permute(2, 0, 1).unsqueeze(0), size=(w_s, h_s), mode="nearest")
            #         .long()
            #         .squeeze(0)
            #         .permute(1, 2, 0)
            #     )
            # concat pred and gt for better visualization
            fig = torch.cat((p, g), 0)

            self.trainer.logger.experiment.add_image(
                "Example_Prediction/prediction_gt__sample_" + str(batch_idx * batch_size + i),
                fig,
                self.current_epoch,
                dataformats="HWC",
            )
