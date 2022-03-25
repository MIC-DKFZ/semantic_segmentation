from omegaconf import OmegaConf
import os
import hydra

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from utils.loss_function import get_loss_function_from_cfg
from utils.utils import hasNotEmptyAttr,hasTrueAttr, hasNotEmptyAttr_rec

from utils.utils import get_logger

log = get_logger(__name__)

class SegModel(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        ### INSTANTIATE MODEL FROM CONFIG ###
        self.model = hydra.utils.instantiate(self.config.model)

        ### INSTANTIATE MEDRIC FROM CONFIG AND METRIC RELATED PARAMETERS ###
        self.metric_name = config.METRIC.NAME

        self.metric_val = hydra.utils.instantiate(config.metric)
        self.register_buffer("best_metric_val", torch.as_tensor(0))

        if hasNotEmptyAttr(config.METRIC,"DURING_TRAIN"):
            self.metric_train = hydra.utils.instantiate(config.metric)
            self.register_buffer("best_metric_train", torch.as_tensor(0))

    def configure_optimizers(self):
        ### INSTANTIATE LOSSFUNCTION AND LOSSWEIGHT FOR EACH ELEMENT IN LIST ###
        if isinstance(self.config.lossfunction,str):
            self.loss_functions = [get_loss_function_from_cfg(self.config.lossfunction, self.config)]
        else:
            self.loss_functions = [get_loss_function_from_cfg(LF, self.config) for LF in self.config.lossfunction]

        if hasattr(self.config, 'lossweight'):
            self.loss_weights = self.config.lossweight
        else:
            self.loss_weights = [1] * len(self.loss_functions)

        #log.info("Loss Functions: %s", self.loss_functions)
        #log.info("Weighting: %s", self.loss_weights)

        #### INSTANTIATE OPTIMIZER ####
        self.optimizer=hydra.utils.instantiate(self.config.optimizer,self.parameters())

        #### INSTANTIATE LR SCHEDULER ####
        max_steps = self.trainer.datamodule.max_steps()

        lr_scheduler_config=dict(self.config.lr_scheduler)
        lr_scheduler_config["scheduler"]=hydra.utils.instantiate(self.config.lr_scheduler.scheduler,
                                                                 optimizer=self.optimizer,
                                                                 max_steps=max_steps)

        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, x):

        x = self.model(x)

        ####COVERT TO DICT IF OUTPUT IS LIST OR TENSOR####
        if not isinstance(x, dict):
            if isinstance(x, list) or isinstance(x, tuple):
                keys=["out"+str(i) for i in range(len(x))]
                x = dict(zip(keys, x))
            elif isinstance(x, torch.Tensor):
                x = {"out": x}
            elif isinstance(x, tuple):

                keys=list(range(0, len(x)))
                x=dict(zip(keys,x))
        return x


    def on_train_epoch_start(self) -> None:
        ### ENSURE METRIC IS RESET AT BEGIN OF EACH EPOCH ###
        if hasattr(self, "metric_train"):
            self.metric_train.reset()

    def training_step(self, batch, batch_idx):

        x, y_gt = batch
        y_pred = self(x)

        loss = self.get_loss(y_pred, y_gt)
        self.log("Loss/training_loss", loss, on_step=True, on_epoch=True, logger=True)#,prog_bar=True)

        if hasattr(self, "metric_train"):
            self.metric_train(y_gt, list(y_pred.values())[0])

        return loss


    def on_validation_epoch_start(self) -> None:
        ### ENSURE METRIC IS RESET AT BEGIN OF EACH EPOCH ###
        self.metric_val.reset()

    def validation_step(self, batch, batch_idx):

        x, y_gt = batch

        y_pred = self(x)

        val_loss = self.get_loss(y_pred, y_gt)
        self.log("Loss/validation_loss", val_loss, on_step=True, on_epoch=True, logger=True)#,prog_bar=True)

        self.metric_val(y_gt, list(y_pred.values())[0])
        return val_loss

    def on_validation_epoch_end(self) -> None:

        if not self.trainer.sanity_checking:
            log.info("EPOCH: %s", self.current_epoch)
            if hasattr(self, "metric_train"):
                self.log_metric(self.metric_train, "best_metric_train", "Train", "metric_train/")

            self.log_metric(self.metric_val, "best_metric_val", "Validation", "metric/",True)



    def test_step(self, batch, batch_idx):

        x, y_gt = batch
        x_size = x.size(2), x.size(3)
        total_pred = None

        ### SET THE DIFFERENT SCALES; IF NO MS TESTING IS USED ONLY SCALE 1 IS USED ###
        if hasNotEmptyAttr(self.config.TESTING,"SCALES"):
            scales=self.config.TESTING.SCALES
        else:
            scales=[1]

        ### ITERATE THROUGH THE SCALES AND SUM THE PREDICTIONS UP ###
        for scale in scales:
            s_size=int(x_size[0]*scale),int(x_size[1]*scale)
            x_s=F.interpolate(x, s_size, mode='bilinear', align_corners=self.config.MODEL.ALIGN_CORNERS)

            y_pred=self(x_s)["out"]
            y_pred=F.interpolate(y_pred, x_size, mode='bilinear', align_corners=self.config.MODEL.ALIGN_CORNERS)

            ###IF FLIPPING IS USED THE AVERAGE OVER PREDICTION FROM THE FLIPPED AND NOT FLIPPEND IMAGE IS TAKEN ###
            if hasTrueAttr(self.config.TESTING,"FLIP"):
                x_flip=torch.flip(x_s, [3])

                y_flip = self(x_flip)["out"]
                y_flip = torch.flip(y_flip, [3])
                y_flip = F.interpolate(y_flip, x_size, mode='bilinear',
                                       align_corners=self.config.MODEL.ALIGN_CORNERS)
                y_pred+=y_flip
                y_pred/=2

            ### SUMMING THE PREDICTIONS UP ###
            if total_pred==None:
                total_pred=y_pred
            else:
                total_pred+=y_pred
        ### UPDATE THE METRIC WITH THE AGGREGATED PREDICTION ###
        self.metric.update(y_gt, total_pred)

    def on_test_epoch_end(self) -> None:
        ### COMPUTE THE METRIC AND LOG THE METRIC ###
        log.info("TEST RESULTS")
        self.log_metric(self.metric, None, "Test", "metric/", False)



    def get_loss(self, y_pred, y_gt):
        ### COMPUTING LOSS OF EVERY INPUT AND THE CORRESPONDING WEIGHT   ###
        ### DURING VALIDATION ONLY CE LOSS IS USED FOR RUNTIME REDUCTION ###
        if self.training:
            loss = sum([self.loss_functions[i](y, y_gt) * self.loss_weights[i] for i, y in enumerate(y_pred.values())])
        else:
            loss = sum([F.cross_entropy(y, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX) * self.loss_weights[i] for i, y in enumerate(y_pred.values())])
        return loss

    def log_metric(self,metric,best_metric=None,stage="Validation",log_group="metric/",save_best_metric=False):
        stage=stage.ljust(10)
        results = metric.compute()

        ### compute and unpack the metrics results ###
        if isinstance(results, tuple):
            metric_score = results[0]  # METRIC TO OPTIMIZE
            metrics_dict = results[1]  # ADDITIONAL METRICS TO LOG
        else:
            metric_score = results
            metrics_dict = None

        if best_metric != None and metric_score > getattr(self,best_metric):
                setattr(self,best_metric,metric_score)
                if hasattr(metric, "save") and save_best_metric:
                    metric.save(path=self.logger.log_dir)

        self.log_dict(
            {log_group + self.metric_name: metric_score, "step": torch.tensor(self.current_epoch, dtype=torch.float32)},
            on_epoch=True, logger=True, sync_dist=True)

        if best_metric == None:
            log.info("%s %s: %.4f",stage.ljust(10), self.metric_name, metric_score)
        else:
            ### LOG BEST METRIC IF EXISTS ###
            self.log_dict(
                {log_group+"best_" + self.metric_name: getattr(self,best_metric),
                 "step": torch.tensor(self.current_epoch, dtype=torch.float32)},
                on_epoch=True, logger=True, sync_dist=True)
            log.info(stage.ljust(10)+" - Best %s %.4f       %s: %.4f", self.metric_name, getattr(self,best_metric), self.metric_name, metric_score)

        ### LOGGING ADDITIONAL METRICS WHICH ARE GIVEN AS DICT ###
        if metrics_dict != None:
            for key in metrics_dict.keys():
                self.log_dict(
                    {"metric_" + self.metric_name + "/" + key: metrics_dict[key].detach(),
                     "step": torch.tensor(self.current_epoch, dtype=torch.float32)},
                    on_epoch=True, logger=True, sync_dist=True)
                log.info("%s: %.4f", self.metric_name + " " + key, metrics_dict[key].detach())




