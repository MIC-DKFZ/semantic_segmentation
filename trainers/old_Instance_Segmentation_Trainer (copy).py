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
import numpy as np
import cv2

log = get_logger(__name__)


class SegModel(LightningModule):
    def __init__(self, config: DictConfig) -> None:
        """
        __init__ the LightningModule
        instantiate the model and the metric(s)

        Parameters
        ----------
        config : omegaconf.DictConfig
        """
        super().__init__()
        self.config = config

        # instantiate model from config
        self.model = hydra.utils.instantiate(self.config.model)
        # print(self.model)
        # print(self.model)
        # quit()
        # instantiate metric from config and metric related parameters
        self.metric_name = config.METRIC.NAME
        # When to Call the Metric
        self.metric_call_global = config.METRIC.call_global
        self.metric_call_stepwise = config.METRIC.call_stepwise
        self.metric_call_per_img = config.METRIC.call_per_img
        # instantiate validation metric from config and save best_metric parameter
        self.metric = MetricModule(config.METRIC.METRICS)  # ,persistent=False)
        self.register_buffer("best_metric_val", torch.as_tensor(0), persistent=False)

        # instantiate train metric from config and save best_metric parameter if wanted
        self.train_metric = config.METRIC.train_metric
        if self.train_metric:
            self.metric_train = self.metric.clone()
            self.register_buffer("best_metric_train", torch.as_tensor(0), persistent=False)

        # create colormap for visualizing the example predictions and also define number of example predictions
        self.cmap = torch.tensor(
            cm.get_cmap("viridis", self.config.DATASET.NUM_CLASSES).colors * 255,
            dtype=torch.uint8,
        )[:, 0:3]
        self.num_example_predictions = (
            config.num_example_predictions
            if has_not_empty_attr(config, "num_example_predictions")
            else 0
        )

    def configure_optimizers(self) -> dict:
        """
        Instantiate the lossfunction + lossweights from the config
        Instantiate the optimizer from the config
        Instantiate the lr scheduler form the config

        Returns
        -------
        dict :
            contains the optimizer and the scheduler + config

        """

        # instantiate lossfunction and lossweight for each element in list
        if isinstance(self.config.lossfunction, str):
            self.loss_functions = [
                get_loss_function_from_cfg(self.config.lossfunction, self.config)
            ]
        else:
            self.loss_functions = [
                get_loss_function_from_cfg(LF, self.config) for LF in self.config.lossfunction
            ]

        if hasattr(self.config, "lossweight"):
            self.loss_weights = self.config.lossweight
        else:
            self.loss_weights = [1] * len(self.loss_functions)
        log.info(
            "Loss Functions with Weights: %s",
            list(zip(self.loss_functions, self.loss_weights)),
        )

        # instantiate optimizer
        params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = hydra.utils.instantiate(self.config.optimizer, params)

        # instantiate lr scheduler
        max_steps = self.trainer.datamodule.max_steps()

        lr_scheduler_config = dict(self.config.lr_scheduler)
        lr_scheduler_config["scheduler"] = hydra.utils.instantiate(
            self.config.lr_scheduler.scheduler,
            optimizer=self.optimizer,
            max_steps=max_steps,
        )

        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, x: torch.Tensor, gt=None) -> dict:
        """
        forward the input to the model
        if model prediction is not a dict covert the output into a dict

        Parameters
        ----------
        x : torch.Tensor
            input to predict

        Returns
        -------
        dict of {str:torch.Tensor} :
            prediction of the model which a separate key for each model output
        """
        if self.training:
            x = self.model(x, gt)
        else:
            x = self.model(x)
        # covert output to dict if output is list, tuple or tensor
        # if not isinstance(x, dict):
        #     if isinstance(x, list) or isinstance(x, tuple):
        #         keys = ["out" + str(i) for i in range(len(x))]
        #         x = dict(zip(keys, x))
        #     elif isinstance(x, torch.Tensor):
        #         x = {"out": x}
        # if torch.isnan(x["out"]).any():
        #     print("NAN Predicted")
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
        # loss = self.get_loss(y_pred, y_gt)
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
        x, y_gt = batch
        y_pred = self(x)
        # print(len(y_pred))
        # y_pred = [{k: v.detach().cpu() for k, v in t.items()} for t in y_pred]
        # y_pred = [round_prediction(y_pred_i) for y_pred_i in y_pred]
        # print(len(y_pred))
        # quit()
        # y_gt = [{k: v.detach().cpu() for k, v in t.items()} for t in y_gt]
        # compute and log loss to tensorboard
        # val_loss = self.get_loss(y_pred, y_gt)
        # self.log(
        #     name="Loss/validation_loss",
        #     value=val_loss,
        #     on_step=True,
        #     on_epoch=True,
        #     logger=True,
        #     sync_dist=True if self.trainer.num_devices > 1 else False,
        # )  # ,prog_bar=True)

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

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metric to logger and console
        Reset the validation metric
        """
        if not self.trainer.sanity_checking:
            log.info("EPOCH: %s", self.current_epoch)

            # compute and log global validation metric to tensorboard
            # if self.metric_call in ["global", "global_and_stepwise"]:
            if self.metric_call_global:
                metric = self.metric.compute()
                self.log_dict_epoch(metric, prefix="metric/", on_step=False, on_epoch=True)

            # log validation metric to console
            self.metric_logger(
                metric_group="metric/",
                best_metric="best_metric_val",
                stage="Validation",
                save_metric_state=True,
            )

        # reset metric manually
        self.metric.reset()

    def on_train_epoch_end(self) -> None:
        """
        (optional) Log the training metric to logger and console
        (optional) Reset the training metric
        """
        # (optional) compute and log global validation metric to tensorboard
        if hasattr(self, "metric_train"):
            if self.metric_call_global:
                metric_train = self.metric_train.compute()

                # log train metric to tensorboard
                self.log_dict_epoch(
                    metric_train, prefix="metric_train/", on_step=False, on_epoch=True
                )
                # reset metric manually

            # log train metric to console
            self.metric_logger(
                metric_group="metric_train/",
                best_metric="best_metric_train",
                stage="Train",
                save_metric_state=False,
            )
            self.metric_train.reset()

    def on_test_start(self) -> None:
        """
        Set the different scales, if no ms testing is used only scale 1 is used
        if not defined also no flipping is done
        """
        self.test_scales = [1]
        self.test_flip = False
        if has_not_empty_attr(self.config, "TESTING"):
            if has_not_empty_attr(self.config.TESTING, "SCALES"):
                self.test_scales = self.config.TESTING.SCALES
            if has_true_attr(self.config.TESTING, "FLIP"):
                self.test_flip = True

    def test_step(self, batch: list, batch_idx: int) -> torch.Tensor:
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
        x, y_gt = batch
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
        self.update_metric(total_pred, y_gt, self.metric, prefix="metric_test/")
        # if self.metric_call in ["stepwise", "global_and_stepwise"]:
        #     # update global metric and log stepwise metric to tensorboard
        #     metric_step = self.metric(total_pred, y_gt)
        #     self.log_dict_epoch(
        #         metric_step,
        #         prefix="metric_test/",
        #         postfix="_stepwise",
        #         on_step=False,
        #         on_epoch=True,
        #     )
        # elif self.metric_call in ["global"]:
        #     # update only global metric
        #     self.metric.update(total_pred, y_gt)

        # compute and return loss of final prediction
        test_loss = F.cross_entropy(total_pred, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX)
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

    def on_test_epoch_end(self) -> None:
        # compute the metric and log the metric
        log.info("TEST RESULTS")

        # compute and log global validation metric to tensorboard
        metric = self.metric.compute()
        self.log_dict_epoch(metric, prefix="metric_test/", on_step=False, on_epoch=True)

        # log validation metric to console
        self.metric_logger(
            metric_group="metric_test/",
            best_metric="best_metric_val",
            stage="Test",
            save_metric_state=True,
        )

        # reset metric manually
        self.metric.reset()

    def update_metric(
        self, y_pred: torch.Tensor, y_gt: torch.Tensor, metric: MetricModule, prefix: str = ""
    ):
        if self.metric_call_stepwise:
            # Log the metric result for each step
            metric_step = metric(y_pred, y_gt)
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
            for yi_pred, yi_gt in zip(y_pred, y_gt):
                metric_step = metric(yi_pred.unsqueeze(0), yi_gt.unsqueeze(0))
                self.log_dict_epoch(
                    metric_step,
                    prefix=prefix,
                    postfix="_per_img",
                    on_step=False,
                    on_epoch=True,
                )
        elif self.metric_call_global:
            # Just update the metric
            metric.update(y_pred, y_gt)

    def get_loss(self, y_pred: dict, y_gt: torch.Tensor) -> torch.Tensor:
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
        return loss

    def metric_logger(self, metric_group, best_metric, stage="Validation", save_metric_state=False):
        """
        logging the metric by:
        update the best metric
        (optional) save state dict variables if provided by the metrics (currently not used)
        log best metric to tensorboard
        log target metric and best metric to console
        log remaining metrics to console

        Parameters
        ----------
        metric_group : str
            enables to group parameters in tensorboard, e.g. into metric/
        best_metric : str
            name of the best metric which corresponds to the target metric
        stage : str, optional
            Current stage, needed for nicer logging
        save_matric_state : bool, optional
            if the metric_state should be saved, currently not used
        """
        logged_metrics = self.trainer.logged_metrics
        metrics = {
            k.replace(metric_group, ""): v for k, v in logged_metrics.items() if metric_group in k
        }

        # update best target metric
        target_metric_score = metrics.pop(self.metric_name)
        if target_metric_score > getattr(self, best_metric):
            setattr(self, best_metric, target_metric_score)

        # (optional) save state of metrics if wanted and provided by the metric, only on rank 0
        if save_metric_state and self.global_rank == 0:
            for name, met in self.metric.items():
                if hasattr(met, "save_state"):
                    met.save_state(self.trainer)

        # log best metric to tensorboard
        if "best_" + self.metric_name in metrics:
            metrics.pop("best_" + self.metric_name)
        self.log_dict_epoch(
            {self.metric_name: getattr(self, best_metric)},
            prefix=metric_group + "best_",
        )
        # log target metric and best metric to console
        log.info(
            stage.ljust(10) + " - Best %s %.4f       %s: %.4f",
            self.metric_name,
            getattr(self, best_metric),
            self.metric_name,
            target_metric_score,
        )
        # remove best_metric from metrics since best metric is already logged to console
        if "best_" + self.metric_name in metrics:
            metrics.pop("best_" + self.metric_name)

        # log remaining metrics to console
        for name, score in metrics.items():
            log.info("%s: %.4f", name, score)

    def log_dict_epoch(self, dic: dict, prefix: str = "", postfix: str = "", **kwargs) -> None:
        """
        Logging a dict to Tensorboard logger but instead of the current step use the current epoch
        on the x-axis of the graphs

        Parameters
        ----------
        dic : dict of {str, torch.Tensor}
            each item contains the name and the scalar for the parameter to log
        prefix : str, optional
            prefix which is added to the name of all logged parameters
        postfix: str, optional
            postfix which is added to the name of all logged parameters
        kwargs: optional
            Parameters to pass to self.log_dict
        """
        for name, score in dic.items():
            self.log_dict(
                {
                    prefix + name + postfix: score,
                    "step": torch.tensor(self.current_epoch, dtype=torch.float32),
                },
                logger=True,
                sync_dist=True if self.trainer.num_devices > 1 else False,
                **kwargs
            )

    def log_batch_prediction(
        self,
        img: torch.Tensor,
        pred: torch.Tensor,
        gt: torch.Tensor,
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

        batch_size = len(img)
        for i in range(min(batch_size, max_number)):

            p = pred[i]
            g = gt[i]
            im = img[i]
            g = show_data(im, g)  # gt.detach().cpu()
            p = show_prediction(im, p)  # pred.argmax(1).detach().cpu()
            p = torch.tensor(p)
            g = torch.tensor(g)
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
            # self.trainer.logger.experiment.add_image(
            #     "Example_Prediction/p" + str(batch_idx * batch_size + i),
            #     p,
            #     self.current_epoch,
            #     dataformats="HWC",
            # )
            # self.trainer.logger.experiment.add_image(
            #     "Example_Prediction/g" + str(batch_idx * batch_size + i),
            #     g,
            #     self.current_epoch,
            #     dataformats="HWC",
            # )
