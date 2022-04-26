import hydra

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from utils.metric import MetricModule
from utils.loss_function import get_loss_function_from_cfg
from utils.utils import hasNotEmptyAttr, hasTrueAttr
from utils.utils import get_logger

log = get_logger(__name__)


class SegModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # instantiate model from config
        self.model = hydra.utils.instantiate(self.config.model)

        # instantiate metric from config and metric related parameters
        self.metric_name = config.METRIC.NAME
        # instantiate validation metric from config and save best_metric parameter
        self.metric = MetricModule(config.METRIC.METRICS)  # ,persistent=False)
        self.register_buffer("best_metric_val", torch.as_tensor(0), persistent=False)

        # define when metric should be called
        self.metric_call = (
            config.METRIC.METRIC_CALL if hasNotEmptyAttr(config.METRIC, "METRIC_CALL") else "global"
        )
        if not self.metric_call in ["global", "stepwise", "global_and_stepwise"]:
            log.warning(
                "Metric Call %s is not in [global,stepwise,global_and_stepwise]: Metric Call will be set to global",
                self.metric_call,
            )
            self.metric_call = "global"

        # (optional) instantiate training metric from config and save best_metric parameter
        if hasTrueAttr(config.METRIC, "DURING_TRAIN"):
            self.metric_train = self.metric.clone()
            self.register_buffer("best_metric_train", torch.as_tensor(0), persistent=False)

    def configure_optimizers(self):
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
        self.optimizer = hydra.utils.instantiate(self.config.optimizer, self.parameters())

        # instantiate lr scheduler
        max_steps = self.trainer.datamodule.max_steps()

        lr_scheduler_config = dict(self.config.lr_scheduler)
        lr_scheduler_config["scheduler"] = hydra.utils.instantiate(
            self.config.lr_scheduler.scheduler,
            optimizer=self.optimizer,
            max_steps=max_steps,
        )

        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, x):

        x = self.model(x)

        # covert output to dict if output is list, tuple or tensor
        if not isinstance(x, dict):
            if isinstance(x, list) or isinstance(x, tuple):
                keys = ["out" + str(i) for i in range(len(x))]
                x = dict(zip(keys, x))
            elif isinstance(x, torch.Tensor):
                x = {"out": x}

        return x

    def training_step(self, batch, batch_idx):
        # predict batch
        x, y_gt = batch
        y_pred = self(x)

        # compute and log loss
        loss = self.get_loss(y_pred, y_gt)
        self.log("Loss/training_loss", loss, on_step=True, on_epoch=True, logger=True)

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

    def validation_step(self, batch, batch_idx):

        # predict batch
        x, y_gt = batch
        y_pred = self(x)

        # compute and log loss to tensorboard
        val_loss = self.get_loss(y_pred, y_gt)
        self.log(
            "Loss/validation_loss", val_loss, on_step=True, on_epoch=True, logger=True
        )  # ,prog_bar=True)

        # update validation metric
        if self.metric_call in ["stepwise", "global_and_stepwise"]:
            # update global metric and log stepwise metric to tensorboard
            metric_step = self.metric(list(y_pred.values())[0], y_gt)
            self.log_dict_epoch(
                metric_step,
                prefix="metric/",
                postfix="_stepwise",
                on_step=False,
                on_epoch=True,
            )
        elif self.metric_call in ["global"]:
            # update only global metric
            self.metric.update(list(y_pred.values())[0], y_gt)

        return val_loss

    def on_validation_epoch_end(self):

        if not self.trainer.sanity_checking:

            log.info("EPOCH: %s", self.current_epoch)

            # compute and log global validation metric to tensorboard
            metric = self.metric.compute()
            self.log_dict_epoch(metric, prefix="metric/", on_step=False, on_epoch=True)

            # log validation metric to console
            self.metric_logger(
                metric_group="metric/",
                best_metric="best_metric_val",
                stage="Validation",
            )

        # reset metric manually
        self.metric.reset()

    def on_train_epoch_end(self):

        # (optional) compute and log global validation metric to tensorboard
        if hasattr(self, "metric_train"):
            metric_train = self.metric_train.compute()

            # log train metric to tensorboard
            self.log_dict_epoch(metric_train, prefix="metric_train/", on_step=False, on_epoch=True)
            # reset metric manually
            self.metric_train.reset()

            # log train metric to console
            self.metric_logger(
                metric_group="metric_train/",
                best_metric="best_metric_train",
                stage="Train",
            )

    def on_test_start(self):

        # set the different scales; if no ms testing is used only scale 1 is used
        # if not defined also no flipping is done
        self.test_scales = [1]
        self.test_flip = False
        if hasNotEmptyAttr(self.config, "TESTING"):
            if hasNotEmptyAttr(self.config.TESTING, "SCALES"):
                self.test_scales = self.config.TESTING.SCALES
            if hasTrueAttr(self.config.TESTING, "FLIP"):
                self.test_flip = True

    def test_step(self, batch, batch_idx):
        x, y_gt = batch
        x_size = x.size(2), x.size(3)
        total_pred = None

        # iterate through the scales and sum the predictions up
        for scale in self.test_scales:
            s_size = int(x_size[0] * scale), int(x_size[1] * scale)
            x_s = F.interpolate(
                x, s_size, mode="bilinear", align_corners=True
            )  # self.config.MODEL.ALIGN_CORNERS)
            y_pred = self(x_s)["out"]
            y_pred = F.interpolate(
                y_pred, x_size, mode="bilinear", align_corners=True
            )  # =self.config.MODEL.ALIGN_CORNERS)

            # if flipping is used the average over the predictions from the flipped and not flipped image is taken
            if self.test_flip:
                print("flip")
                x_flip = torch.flip(x_s, [3])  #

                y_flip = self(x_flip)["out"]
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
            metric_step = self.metric(y_pred, y_gt)
            self.log_dict_epoch(
                metric_step,
                prefix="metric_test/",
                postfix="_stepwise",
                on_step=False,
                on_epoch=True,
            )
        elif self.metric_call in ["global"]:
            # update only global metric
            self.metric.update(y_pred, y_gt)

    def on_test_epoch_end(self):
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
        )

        # reset metric manually
        self.metric.reset()

    def metric_logger(self, metric_group, best_metric=None, stage="Validation"):
        logged_metrics = self.trainer.logged_metrics

        metrics = {
            k.replace(metric_group, ""): v for k, v in logged_metrics.items() if metric_group in k
        }

        # update best target metric
        target_metric_score = metrics.pop(self.metric_name)
        if target_metric_score > getattr(self, best_metric):
            setattr(self, best_metric, target_metric_score)

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
        # remove best metric from metrics since best metric is already logged to console
        if "best_" + self.metric_name in metrics:
            metrics.pop("best_" + self.metric_name)

        # log remaining metrics to console
        for name, score in metrics.items():
            log.info("%s: %.4f", name, score)

    def log_dict_epoch(self, dic, prefix="", postfix="", **kwargs):
        for name, score in dic.items():
            self.log_dict(
                {
                    prefix + name + postfix: score,
                    "step": torch.tensor(self.current_epoch, dtype=torch.float32),
                },
                logger=True,
                sync_dist=True,
                **kwargs
            )

    def get_loss(self, y_pred, y_gt):
        # computing loss of every output and the corresponding weight
        # during validation only ce loss is used for runtime reduction
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
