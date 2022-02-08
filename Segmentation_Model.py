import logging
from omegaconf import OmegaConf
import os
import hydra

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from pytorch_lightning import LightningModule

from utils.loss_function import get_loss_function_from_cfg
from utils.optimizer import get_optimizer_from_cfg
from utils.lr_scheduler import get_lr_scheduler_from_cfg
from utils.utils import hasNotEmptyAttr,hasTrueAttr

from models import hrnet, hrnet_ocr, hrnet_ocr_aspp, hrnet_ocr_ms

log = logging.getLogger(__name__)

class ConfusionMatrix(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.num_classes = num_classes
        self.add_state("mat", default=torch.zeros((num_classes, num_classes), dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, gt, pred):

        n = self.num_classes
        gt=gt.detach().cpu()
        pred=pred.detach().cpu()

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n).to(self.mat)

    def compute(self):
        IoU = self.mat.diag() / (self.mat.sum(1) + self.mat.sum(0) - self.mat.diag())
        IoU[IoU.isnan()] = 0
        mIoU = IoU.mean()
        return IoU, mIoU

    def save(self, path):
        path = os.path.join(path, "ConfusionMatrix.pt")
        torch.save(self.mat.cpu(), path)

    def save_named(self, path, name):
        path = os.path.join(path, "ConfusionMatrix_"+name+".pt")
        torch.save(self.mat.cpu(), path)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

class SegModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        #self.num_classes=config.DATASET.NUM_CLASSES
        self.model = eval(config.MODEL.NAME + '.get_seg_model')(config)

        self.metric = ConfusionMatrix(config.DATASET.NUM_CLASSES)
        self.register_buffer("best_mIoU", torch.as_tensor(0))

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

        log.info("Loss Functions: %s", self.loss_functions)
        log.info("Weighting: %s", self.loss_weights)

        #### OPTIMIZER ####
        #self.optimizer = get_optimizer_from_cfg(self.parameters(), self.config)
        self.optimizer=hydra.utils.instantiate(self.config.optimizer,self.parameters())

        #### LR SCHEDULER ####
        max_steps = self.trainer.datamodule.max_steps()


        #self.lr_scheduler, interval =hydra.utils.instantiate(self.config.lr_scheduler,optimizer=self.optimizer,max_steps=max_steps)
        #lr_scheduler_config= {"scheduler": self.lr_scheduler, 'interval': interval, 'frequency': 1,
        #                   "monitor": "metric_to_track"}

        lr_scheduler_config=dict(self.config.lr_scheduler_config)
        lr_scheduler_config["scheduler"]=hydra.utils.instantiate(self.config.lr_scheduler_config.scheduler,
                                                                 optimizer=self.optimizer,
                                                                 max_steps=max_steps)

        #lr_scheduler_config = hydra.utils.instantiate(self.config.lr_scheduler_config,optimizer=self.optimizer, max_steps=self.trainer.datamodule.max_steps())
        #lr_scheduler_config = self.config.lr_scheduler_config
        #lr_scheduler_config.scheduler=hydra.utils.call(lr_scheduler_config.scheduler,optimizer=self.optimizer,max_steps=max_steps)
        #### LR SCHEDULER ####
        #max_steps = self.trainer.datamodule.max_steps()
        #self.lr_scheduler, lr_scheduler_config = get_lr_scheduler_from_cfg(self.optimizer, max_steps, self.config)

        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def on_train_start(self):
        # only used for for logging the hyperparameters and the metrics (IoU and Time)
        num_total = sum(p.numel() for p in self.model.parameters())
        num_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.log_hyperparams(
            {"Pretrained": self.config.MODEL.PRETRAINED, "Parameter": num_total, "trainable Parameter": num_train},
            {"mIoU/best_mIoU": self.best_mIoU, "Time/mTrainTime": 0, "Time/mValTime": 0})
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

        self.metric.update(y_gt.flatten(), list(y_pred.values())[0].argmax(1).flatten())

        return val_loss

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:

            IoU, mIoU = self.metric.compute()
            self.log_dict({"mIoU": mIoU,"step":self.current_epoch}, on_epoch=True, logger=True, sync_dist=True)
            #self.log("mIoU", mIoU, logger=True, sync_dist=True)

            if mIoU > self.best_mIoU.item():
                self.best_mIoU = mIoU
                self.metric.save(path=self.logger.log_dir)
            self.log_dict({"mIoU/best_mIoU": self.best_mIoU, "step": self.current_epoch}, on_epoch=True, logger=True,sync_dist=True)
            #self.log("mIoU/best_mIoU", self.best_mIoU, logger=True, sync_dist=True)

            dic_IoU = {}
            for id, iou in enumerate(IoU):
                if hasNotEmptyAttr(self.config.DATASET,"CLASS_LABELS"):
                    dic_IoU[str(id) + "-" + self.config.DATASET.CLASS_LABELS[id]] = "%.4f" % iou.item()
                else:
                    dic_IoU[id] = "%.4f" % iou.item()

            #if self.trainer.is_global_zero:
            log.info("EPOCH: %s", self.current_epoch)
            log.info("Best mIoU %.4f      Mean IoU: %.4f", self.best_mIoU, mIoU.item())
            log.info(dic_IoU)

    def test_step(self, batch, batch_idx):

        x, y_gt = batch
        x_size = x.size(2), x.size(3)
        total_pred = None

        if hasNotEmptyAttr(self.config.TESTING,"SCALES") and hasTrueAttr(self.config.TESTING,"MS_TESTING") :
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
        self.metric.update(y_gt.flatten(), total_pred.argmax(1).flatten())

    def on_test_epoch_end(self) -> None:

        IoU, mIoU = self.metric.compute()

        dic_IoU = {}
        for id, iou in enumerate(IoU):
            if hasNotEmptyAttr(self.config.DATASET,"CLASS_LABELS"):
                dic_IoU[str(id) + "-" + self.config.DATASET.CLASS_LABELS[id]] = "%.4f" % iou.item()
            else:
                dic_IoU[id] = "%.4f" % iou.item()

        #if self.trainer.is_global_zero:
        log.info("EPOCH: %s", self.current_epoch)
        log.info(dic_IoU)
        log.info("Best mIoU %.4f", mIoU.item())

        self.metric.save_named(path=self.logger.log_dir,name="%.4f" % mIoU.item())


