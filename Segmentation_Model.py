from omegaconf import OmegaConf
import os
import hydra

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from utils.loss_function import get_loss_function_from_cfg
from utils.utils import hasNotEmptyAttr,hasTrueAttr

from utils.utils import get_logger
log = get_logger(__name__)

class SegModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = hydra.utils.instantiate(self.config.model)

        self.metric = hydra.utils.instantiate(config.metric)
        if hasNotEmptyAttr(config.METRIC,"NAME"): self.metric_name=config.METRIC.NAME
        else: self.metric_name="metric"
        self.register_buffer("best_metric_score", torch.as_tensor(0))

    def forward(self, x):

        x = self.model(x)

        return x

    def configure_optimizers(self):
        #### LOSSFUNCTION ####
        if isinstance(self.config.lossfunction,str):
            self.loss_functions = [get_loss_function_from_cfg(self.config.lossfunction, self.config)]
        else:
            self.loss_functions = [get_loss_function_from_cfg(LF, self.config) for LF in self.config.lossfunction]

        if hasattr(self.config, 'lossweight'):
            self.loss_weights = self.config.lossweight
        else:
            self.loss_weights = [1] * len(self.loss_functions)
        #print(self.trainer.datamodule.)
        log.info("Loss Functions: %s", self.loss_functions)
        log.info("Weighting: %s", self.loss_weights)

        #### OPTIMIZER ####
        self.optimizer=hydra.utils.instantiate(self.config.optimizer,self.parameters())

        #### LR SCHEDULER ####
        max_steps = self.trainer.datamodule.max_steps()

        lr_scheduler_config=dict(self.config.lr_scheduler)
        lr_scheduler_config["scheduler"]=hydra.utils.instantiate(self.config.lr_scheduler.scheduler,
                                                                 optimizer=self.optimizer,
                                                                 max_steps=max_steps)

        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def on_train_start(self):
        # only used for for logging the hyperparameters and the metrics (IoU and Time)
        num_total = sum(p.numel() for p in self.model.parameters())
        num_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.log_hyperparams(
            {"Parameter": num_total, "trainable Parameter": num_train},
            {"metric/best_"+self.metric_name: self.best_metric_score, "Time/mTrainTime": 0, "Time/mValTime": 0})
        #saving resolved parameters
        OmegaConf.save(config=self.config, resolve=True, f=os.path.join(self.logger.log_dir, "hparams.yaml"))

    def get_loss(self, y_pred, y_gt):
        if self.training:
            loss = sum([self.loss_functions[i](y, y_gt) * self.loss_weights[i] for i, y in enumerate(y_pred.values())])
        else:
            ### only use CE loss during Validation for runtime reduction ###
            loss = sum([F.cross_entropy(y, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX) * self.loss_weights[i] for i, y in enumerate(y_pred.values())])
        return loss

    def training_step(self, batch, batch_idx):

        x, y_gt = batch
        y_pred = self(x)

        loss = self.get_loss(y_pred, y_gt)
        self.log("Loss/training_loss", loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y_gt = batch

        y_pred = self(x)

        val_loss = self.get_loss(y_pred, y_gt)
        self.log("Loss/validation_loss", val_loss, on_step=True, on_epoch=True, logger=True)

        self.metric(y_gt, list(y_pred.values())[0])

        return val_loss

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()

    def log_results(self,metric_score,metrics_dict):

        self.log_dict(
            {"metric/" + self.metric_name: metric_score, "step": torch.tensor(self.current_epoch, dtype=torch.float32)},
            on_epoch=True, logger=True, sync_dist=True)
        log.info("Best %s %.4f       %s: %.4f", self.metric_name, self.best_metric_score, self.metric_name,
                 metric_score.detach())

        if metrics_dict!=None:
            for key in metrics_dict.keys():

                self.log_dict(
                    {"metric/" + key: metrics_dict[key].detach(),
                     "step": torch.tensor(self.current_epoch, dtype=torch.float32)},
                    on_epoch=True, logger=True, sync_dist=True)
                log.info("%s: %.4f", key, metrics_dict[key].detach())

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:

            log.info("EPOCH: %s", self.current_epoch)

            results = self.metric.compute()
            if isinstance(results, tuple):
                metric_score = results[0]
                metrics_dict = results[1]
            else:
                metric_score = results
                metrics_dict = None

            if metric_score > self.best_metric_score.detach():
                self.best_metric_score = metric_score
                if hasattr(self.metric, "save"):
                    self.metric.save(path=self.logger.log_dir)

            self.log_results(metric_score,metrics_dict)

    def test_step(self, batch, batch_idx):
        #implementation of multiscale testing
        x, y_gt = batch
        x_size = x.size(2), x.size(3)
        total_pred = None

        if hasNotEmptyAttr(self.config.TESTING,"SCALES"):
            scales=self.config.TESTING.SCALES
        else:
            scales=[1]

        for scale in scales:
            s_size=int(x_size[0]*scale),int(x_size[1]*scale)
            x_s=F.interpolate(x, s_size, mode='bilinear', align_corners=self.config.MODEL.ALIGN_CORNERS)

            y_pred=self(x_s)["out"]
            y_pred=F.interpolate(y_pred, x_size, mode='bilinear', align_corners=self.config.MODEL.ALIGN_CORNERS)

            if hasTrueAttr(self.config.TESTING,"FLIP"):
                x_flip=torch.flip(x_s, [3])

                y_flip = self(x_flip)["out"]
                y_flip = torch.flip(y_flip, [3])
                y_flip = F.interpolate(y_flip, x_size, mode='bilinear',
                                       align_corners=self.config.MODEL.ALIGN_CORNERS)
                y_pred+=y_flip
                y_pred/=2

            if total_pred==None:
                total_pred=y_pred
            else:
                total_pred+=y_pred
        self.metric.update(y_gt, total_pred)

    def on_test_epoch_end(self) -> None:

        log.info("Test Results")
        results = self.metric.compute()
        if isinstance(results, tuple):
            metric_score = results[0]
            metrics_dict = results[1]
        else:
            metric_score = results
            metrics_dict = None

        self.log_results(metric_score, metrics_dict)

        if hasattr(self.metric,"save"):
            self.metric.save(path=self.logger.log_dir,name="%.4f" % metric_score.detach())


