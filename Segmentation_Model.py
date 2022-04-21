from omegaconf import OmegaConf
import os
import hydra

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from utils.metric import MetricModule
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
        ### INSTANTIATE VALIDATION MEDRIC FROM CONFIG AND SAVE BEST METRIC PARAMETER ###
        self.metric=MetricModule(config.METRIC.METRICS)#,persistent=False)
        self.register_buffer("best_metric_val", torch.as_tensor(0),persistent=False)

        ### (OPTIONAL) INSTANTIATE TRAINING MEDRIC FROM CONFIG AND SAVE BEST METRIC PARAMETER ###
        if hasTrueAttr(config.METRIC, "DURING_TRAIN"):
            self.metric_train=self.metric.clone()
            self.register_buffer("best_metric_train", torch.as_tensor(0),persistent=False)


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

        log.info("Loss Functions with Weights: %s", list(zip(self.loss_functions,self.loss_weights)))

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

        ####COVERT OUTPUT TO DICT IF OUTPUT IS LIST, TUPEL OR TENSOR####
        if not isinstance(x, dict):
            if isinstance(x, list) or isinstance(x, tuple):
                keys=["out"+str(i) for i in range(len(x))]
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
            self.metric_train.update(list(y_pred.values())[0],y_gt)

        return loss

    def validation_step(self, batch, batch_idx):

        ### PREDICT BATCH ###
        x, y_gt = batch
        y_pred = self(x)

        ### COMPUTE AND LOG LOSS ###
        val_loss = self.get_loss(y_pred, y_gt)
        self.log("Loss/validation_loss", val_loss, on_step=True, on_epoch=True, logger=True)#,prog_bar=True)

        ### UPDATE VALIDATION METRIC ###
        self.metric.update(list(y_pred.values())[0],y_gt)

        return val_loss

    def on_validation_epoch_end(self):

        if not self.trainer.sanity_checking:

            log.info("EPOCH: %s", self.current_epoch)

            ### COMPUTE, LOG AND RESET THE TRAINING METRICS IF USED ###
            if hasattr(self, "metric_train"):

                metrics_train = self.metric_train.compute()

                self.log_metric(metrics_train, "best_metric_train", "Train", "metric_train/", False)

                self.metric_train.reset()


            ### COMPUTE AND LOG THE VALIDATION METRICS ###
            metrics=self.metric.compute()

            self.log_metric(metrics, "best_metric_val", "Validation", "metric/", True)

        ### RESET METRIC MANUALLY###
        self.metric.reset()

    def on_test_start(self):
        self.test_scales = [1]
        self.test_flip = False
        if hasNotEmptyAttr(self.config,"TESTING"):
            if hasNotEmptyAttr(self.config.TESTING,"SCALES"):
                self.test_scales=self.config.TESTING.SCALES
            if hasTrueAttr(self.config.TESTING, "FLIP"):
                self.test_flip=True



    def test_step(self, batch, batch_idx):
        #print(self.stage)
        #self.validation_step(batch, batch_idx)
        #return
        x, y_gt = batch
        x_size = x.size(2), x.size(3)
        total_pred = None

        ### SET THE DIFFERENT SCALES; IF NO MS TESTING IS USED ONLY SCALE 1 IS USED ###

        #print(scales,do_flip)
        ### ITERATE THROUGH THE SCALES AND SUM THE PREDICTIONS UP ###
        for scale in self.test_scales:
            s_size=int(x_size[0]*scale),int(x_size[1]*scale)
            x_s=F.interpolate(x, s_size, mode='bilinear', align_corners=self.config.MODEL.ALIGN_CORNERS)
            y_pred = self(x_s)["out"]
            #y_pred=self(x_s)["out"]
            y_pred=F.interpolate(y_pred, x_size, mode='bilinear', align_corners=self.config.MODEL.ALIGN_CORNERS)

            ###IF FLIPPING IS USED THE AVERAGE OVER PREDICTION FROM THE FLIPPED AND NOT FLIPPEND IMAGE IS TAKEN ###
            if self.test_flip:
                print("flip")
                x_flip=torch.flip(x_s, [3])#

                y_flip = self(x_flip)["out"]
                y_flip = torch.flip(y_flip, [3])
                y_flip = F.interpolate(y_flip, x_size, mode='bilinear',
                                       align_corners=self.config.MODEL.ALIGN_CORNERS)
                y_pred+=y_flip
                y_pred/=2

            ### SUMMING THE PREDICTIONS UP ###
            if total_pred==None:
                total_pred=y_pred#.detach()
            else:
                total_pred+=y_pred#.detach()
        ### UPDATE THE METRIC WITH THE AGGREGATED PREDICTION ###
        self.metric.update(total_pred,y_gt)

    def on_test_epoch_end(self):
        ### COMPUTE THE METRIC AND LOG THE METRIC ###
        log.info("TEST RESULTS")

        metrics = self.metric.compute()

        self.log_metric(metrics, "best_metric_val", "Test", "metric/", False)



    def get_loss(self, y_pred, y_gt):
        ### COMPUTING LOSS OF EVERY OUTPUT AND THE CORRESPONDING WEIGHT   ###
        ### DURING VALIDATION ONLY CE LOSS IS USED FOR RUNTIME REDUCTION ###
        if self.training:
            loss = sum([self.loss_functions[i](y, y_gt) * self.loss_weights[i] for i, y in enumerate(y_pred.values())])
        else:
            #loss = sum([F.cross_entropy(y, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX)for i, y in enumerate(y_pred.values())])
            loss = sum([F.cross_entropy(y, y_gt, ignore_index=self.config.DATASET.IGNORE_INDEX) * self.loss_weights[i] for i, y in enumerate(y_pred.values())])
        return loss

    def log_metric(self,metrics,best_metric=None,stage="Validation",log_group="metric/",save_best_metric=False):

        ### FIRST LOG TARGET METRIC   ###
        target_metric_score = metrics.pop(self.metric_name)
        ### LOGGING TO TENSORBOARD ###
        self.log_dict({log_group + self.metric_name: target_metric_score,
                       "step": torch.tensor(self.current_epoch, dtype=torch.float32)},
                      logger=True, sync_dist=True)

        ### UPDATA BEST METRIC ###
        if target_metric_score > getattr(self,best_metric):
            setattr(self,best_metric,target_metric_score)
            if hasattr(self.metric[self.metric_name], "save") and save_best_metric and hasattr(self.logger, "log_dir"):
                self.metric[self.metric_name].save(path=self.logger.log_dir)
        self.log_dict({log_group +"best_"+ self.metric_name: target_metric_score,
                       "step": torch.tensor(self.current_epoch, dtype=torch.float32)},
                      logger=True, sync_dist=True)

        ### CONSOLE LOGGING ###
        log.info(stage.ljust(10) + " - Best %s %.4f       %s: %.4f", self.metric_name, getattr(self, best_metric),
                 self.metric_name, target_metric_score)


        ### LOG THE REMAINING METRICS ###
        for name,score in metrics.items():
            ### LOGGING TO TENSORBOARD ###
            self.log_dict({log_group + name: score,"step": torch.tensor(self.current_epoch, dtype=torch.float32)}, logger=True, sync_dist=True)
            ### CONSOLE LOGGING ###
            log.info('%s: %.4f',name,score)






