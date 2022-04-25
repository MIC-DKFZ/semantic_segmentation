from omegaconf import OmegaConf
import os
import hydra

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from utils.metric import MetricModule
from utils.loss_function import get_loss_function_from_cfg
from utils.utils import hasNotEmptyAttr, hasTrueAttr, hasNotEmptyAttr_rec
from utils.utils import get_logger

log = get_logger(__name__)
# model=hrnet epochs=5 +pl_trainer.limit_train_batches=0.1 +pl_trainer.limit_val_batches=0.1 pl_trainer.enable_checkpointing=True
class SegModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ### INSTANTIATE MODEL FROM CONFIG ###
        self.model = hydra.utils.instantiate(self.config.model)

        ### INSTANTIATE MEDRIC FROM CONFIG AND METRIC RELATED PARAMETERS ###
        self.metric_name = config.METRIC.NAME
        ### INSTANTIATE VALIDATION MEDRIC FROM CONFIG AND SAVE BEST METRIC PARAMETER ###
        self.metric = MetricModule(config.METRIC.METRICS)  # ,persistent=False)
        self.metric_call = config.METRIC.METRIC_CALL if hasNotEmptyAttr(config.METRIC,"METRIC_CALL") else "global"
        self.register_buffer("best_metric_val", torch.as_tensor(0), persistent=False)

        ### (OPTIONAL) INSTANTIATE TRAINING MEDRIC FROM CONFIG AND SAVE BEST METRIC PARAMETER ###
        if hasTrueAttr(config.METRIC, "DURING_TRAIN"):
            self.metric_train = self.metric.clone()
            self.register_buffer("best_metric_train", torch.as_tensor(0), persistent=False)

    def configure_optimizers(self):
        ### INSTANTIATE LOSSFUNCTION AND LOSSWEIGHT FOR EACH ELEMENT IN LIST ###
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

        #### INSTANTIATE OPTIMIZER ####
        self.optimizer = hydra.utils.instantiate(self.config.optimizer, self.parameters())

        #### INSTANTIATE LR SCHEDULER ####
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

        ####COVERT OUTPUT TO DICT IF OUTPUT IS LIST, TUPEL OR TENSOR####
        if not isinstance(x, dict):
            if isinstance(x, list) or isinstance(x, tuple):
                keys = ["out" + str(i) for i in range(len(x))]
                x = dict(zip(keys, x))
            elif isinstance(x, torch.Tensor):
                x = {"out": x}

        return x

    def training_step(self, batch, batch_idx):

        ### PREDICT BATCH ###
        x, y_gt = batch
        y_pred = self(x)

        ### COMPUTE AND LOG LOSS ###
        loss = self.get_loss(y_pred, y_gt)
        self.log("Loss/training_loss", loss, on_step=True, on_epoch=True, logger=True)

        ### (OPTIONAL) UPDATE TRAIN METRIC ###
        if hasattr(self, "metric_train"):
            if self.metric_call in ["stepwise"]:
                ### UPDATE GLOBAL METRIC AND LOG STEPWISE METRIC ###
                metric_step = self.metric_train(list(y_pred.values())[0], y_gt)
                self.log_dict_epoch(
                    metric_step,
                    prefix="metric_train/",
                    postfix="_stepwise",
                    on_step=False,
                    on_epoch=True,
                )
            else:
                ### UPDATE ONLY GLOBAL METRIC ###
                self.metric_train.update(list(y_pred.values())[0], y_gt)

        return loss

    def validation_step(self, batch, batch_idx):

        ### PREDICT BATCH ###
        x, y_gt = batch
        y_pred = self(x)

        ### COMPUTE AND LOG LOSS TO TENSORBOARD ###
        val_loss = self.get_loss(y_pred, y_gt)
        self.log(
            "Loss/validation_loss", val_loss, on_step=True, on_epoch=True, logger=True
        )  # ,prog_bar=True)

        ### UPDATE VALIDATION METRIC ###
        if self.metric_call in ["stepwise"]:
            ### UPDATE GLOBAL METRIC AND LOG STEPWISE METRIC TO TENSORBOARD###
            metric_step = self.metric(list(y_pred.values())[0], y_gt)
            self.log_dict_epoch(
                metric_step,
                prefix="metric/",
                postfix="_stepwise",
                on_step=False,
                on_epoch=True,
            )
        else:
            ### UPDATE ONLY GLOBAL METRIC ###
            self.metric.update(list(y_pred.values())[0], y_gt)

        return val_loss

    def on_validation_epoch_end(self):

        if not self.trainer.sanity_checking:

            log.info("EPOCH: %s", self.current_epoch)

            ### COMPUTE AND LOG GLOBAL VALIDATION METRIC TO TENSORBOARD ###
            metric = self.metric.compute()
            self.log_dict_epoch(metric, prefix="metric/", on_step=False, on_epoch=True)

            ### LOG VALIDATION METRIC TO CONSOLE ###
            self.metric_logger(
                metric_group="metric/",
                best_metric="best_metric_val",
                stage="Validation",
                save_best_metric=True,
            )

        ### RESET METRIC MANUALLY###
        self.metric.reset()

    def on_train_epoch_end(self):
        ### (OPTIONAL) COMPUTE AND LOG GLOBAL VALIDATION METRIC TO TENSORBOARD ###
        if hasattr(self, "metric_train"):
            metric_train = self.metric_train.compute()
            self.log_dict_epoch(metric_train, prefix="metric_train/", on_step=False, on_epoch=True)
            ### RESET METRIC MANUALLY###
            self.metric_train.reset()
            ### (OPTIONAL) LOG TRAIN METRIC TO CONSOLE ###
            self.metric_logger(
                metric_group="metric_train/",
                best_metric="best_metric_train",
                stage="Train",
                save_best_metric=False,
            )

    def on_test_start(self):

        ### SET THE DIFFERENT SCALES; IF NO MS TESTING IS USED ONLY SCALE 1 IS USED ###
        ### IF NOT DEFINED ALSO NO FLIPPING IS DONE ###
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

        ### ITERATE THROUGH THE SCALES AND SUM THE PREDICTIONS UP ###
        for scale in self.test_scales:
            s_size = int(x_size[0] * scale), int(x_size[1] * scale)
            x_s = F.interpolate(
                x, s_size, mode="bilinear", align_corners=True
            )  # self.config.MODEL.ALIGN_CORNERS)
            y_pred = self(x_s)["out"]
            y_pred = F.interpolate(
                y_pred, x_size, mode="bilinear", align_corners=True
            )  # =self.config.MODEL.ALIGN_CORNERS)

            ###IF FLIPPING IS USED THE AVERAGE OVER PREDICTION FROM THE FLIPPED AND NOT FLIPPEND IMAGE IS TAKEN ###
            if self.test_flip:
                print("flip")
                x_flip = torch.flip(x_s, [3])  #

                y_flip = self(x_flip)["out"]
                y_flip = torch.flip(y_flip, [3])
                y_flip = F.interpolate(y_flip, x_size, mode="bilinear", align_corners=True)
                # align_corners=self.config.MODEL.ALIGN_CORNERS)
                y_pred += y_flip
                y_pred /= 2

            ### SUMMING THE PREDICTIONS UP ###
            if total_pred == None:
                total_pred = y_pred  # .detach()
            else:
                total_pred += y_pred  # .detach()

        ### UPDATE THE METRIC WITH THE AGGREGATED PREDICTION ###
        if self.metric_call in ["stepwise"]:
            ### UPDATE GLOBAL METRIC AND LOG STEPWISE METRIC TO TENSORBOARD###
            metric_step = self.metric(y_pred, y_gt)
            self.log_dict_epoch(
                metric_step,
                prefix="metric_test/",
                postfix="_stepwise",
                on_step=False,
                on_epoch=True,
            )
        else:
            ### UPDATE ONLY GLOBAL METRIC ###
            self.metric.update(y_pred, y_gt)

    def on_test_epoch_end(self):
        ### COMPUTE THE METRIC AND LOG THE METRIC ###
        log.info("TEST RESULTS")

        ### COMPUTE AND LOG GLOBAL VALIDATION METRIC TO TENSORBOARD ###
        metric = self.metric.compute()
        self.log_dict_epoch(metric, prefix="metric_test/", on_step=False, on_epoch=True)

        ### LOG VALIDATION METRIC TO CONSOLE ###
        self.metric_logger(
            metric_group="metric_test/",
            best_metric="best_metric_val",
            stage="Test",
            save_best_metric=False,
        )

        ### RESET METRIC MANUALLY###
        self.metric.reset()

    def metric_logger(
        self, metric_group, best_metric=None, stage="Validation", save_best_metric=True
    ):
        logged_metrics = self.trainer.logged_metrics

        metrics = {
            k.replace(metric_group, ""): v for k, v in logged_metrics.items() if metric_group in k
        }

        ### UPDATE BEST TARGET METRIC ###
        target_metric_score = metrics.pop(self.metric_name)
        if target_metric_score > getattr(self, best_metric):
            setattr(self, best_metric, target_metric_score)

        ### LOG BEST METRIC TO TENSORBOARD ###
        if "best_" + self.metric_name in metrics:
            metrics.pop("best_" + self.metric_name)
        self.log_dict_epoch(
            {self.metric_name: getattr(self, best_metric)},
            prefix=metric_group + "best_",
        )
        ### LOG TARGET METRIC AND BEST METRIC TO CONSOLE ###
        log.info(
            stage.ljust(10) + " - Best %s %.4f       %s: %.4f",
            self.metric_name,
            getattr(self, best_metric),
            self.metric_name,
            target_metric_score,
        )
        ### REMOVE BEST METRIC FROM metrics since best metric is already logged to console ###
        if "best_" + self.metric_name in metrics:
            metrics.pop("best_" + self.metric_name)

        ### LOG REMAINING METRICS TO CONSOLE ###
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
        ### COMPUTING LOSS OF EVERY OUTPUT AND THE CORRESPONDING WEIGHT   ###
        ### DURING VALIDATION ONLY CE LOSS IS USED FOR RUNTIME REDUCTION ###
        if self.training:
            loss = sum(
                [
                    self.loss_functions[i](y, y_gt) * self.loss_weights[i]
                    for i, y in enumerate(y_pred.values())
                ]
            )
        else:
            # loss = sum([F.cross_entropy(y, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX)for i, y in enumerate(y_pred.values())])
            loss = sum(
                [
                    F.cross_entropy(y, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX)
                    * self.loss_weights[i]
                    for i, y in enumerate(y_pred.values())
                ]
            )
        return loss
