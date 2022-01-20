import glob
import logging
import os.path
import hydra
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import numpy as np
import torch
from torchmetrics import Metric
import torch.nn.functional as F

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin

from config.utils import hasTrueAttr, hasNotEmptyAttr
from omegaconf import DictConfig, OmegaConf
from utils.loss_function import get_loss_function_from_cfg
from utils.optimizer import get_optimizer_from_cfg
from utils.lr_scheduler import get_lr_scheduler_from_cfg

#from datasets import DataModules
from models import hrnet, hrnet_ocr, hrnet_ocr_aspp, hrnet_ocr_ms
#from models import hrnet_ocr3 as hrnet_ocr

#import time

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
        self.optimizer = get_optimizer_from_cfg(self.parameters(), self.config)

        #### LR SCHEDULER ####
        max_steps = self.trainer.datamodule.max_steps()
        self.lr_scheduler, lr_scheduler_config = get_lr_scheduler_from_cfg(self.optimizer, max_steps, self.config)

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

    def get_loss2(self, y_pred, y_gt):
        #print("P",y_pred["out"].shape)
        loss=0
        for i,(w,x) in enumerate(zip(self.loss_weights,y_pred.values())):
            for a,b in zip(x,y_gt):
                #print(a.shape, b.shape)
                s1,s2=b.size()
                a= F.interpolate(a.unsqueeze(0), size=(s1,s2), mode='bilinear', align_corners=True)
                loss+=self.loss_functions[i](a, b.unsqueeze(0)) * self.loss_weights[i]
            #print(w,x.shape)
        #loss = sum([self.loss_functions[i](y, y_gt) * self.loss_weights[i] for i, y in enumerate(y_pred.values())])
        #print("L",loss)
        return loss

    def get_loss(self, y_pred, y_gt):

        loss = sum([self.loss_functions[i](y, y_gt) * self.loss_weights[i] for i, y in enumerate(y_pred.values())])

        return loss

    def training_step(self, batch, batch_idx):

        x, y_gt = batch

        y_pred = self(x)

        loss = self.get_loss(y_pred, y_gt)
        self.log("Loss/training_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        #if len(batch)==3:
        #    x, y_gt, y_org=batch
        #else:
        x, y_gt = batch

        #x,y_gt=batch["img"],batch["mask"]


        #x, y_gt = batch
        #print(batch_idx,idx)
        #self.trainer.datamodule.CS_train.get_idx(batch_idx)
        #print(x[0].shape)
        #x=torch.stack(x,dim=0)

        #print("X",x.shape)
        #print(y_gt)

        y_pred = self(x)
        #print(y_pred[0])
        #print(y_gt[0].shape)


        val_loss = self.get_loss(y_pred, y_gt)
        self.log("Loss/validation_loss", val_loss, on_step=True, on_epoch=True, logger=True)

        #if len(batch)==3:
        #    pred=list(y_pred.values())[0]
        #    size=[x.size() for x in y_org]
        #    pred=[F.interpolate(p.unsqueeze(0), size=s, mode='bilinear', align_corners=False).argmax(1).flatten() for p,s in zip(pred,size)]
        #
        #    pred=torch.cat(pred)
        #    y_org=torch.cat([y.flatten() for y in y_org])#

        #    self.metric.update(y_org, pred)
        #else:

        self.metric.update(y_gt.flatten(), list(y_pred.values())[0].argmax(1).flatten())

        #for p,g in zip(list(y_pred.values())[0],y_gt):
        #    #print("PG",p.shape,g.shape)
        #    s1, s2 = g.size()
        #    p = F.interpolate(p.unsqueeze(0), size=(s1, s2), mode='bilinear', align_corners=True)
            #loss += self.loss_functions[i](a, b.unsqueeze(0)) * self.loss_weights[i]

        #    self.metric.update(g.flatten(), p.argmax(1).flatten())


        return val_loss

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:

            IoU, mIoU = self.metric.compute()
            self.log_dict({"mIoU": mIoU,"step":self.current_epoch}, on_epoch=True, logger=True,sync_dist=True)
            #self.log("mIoU", mIoU, logger=True, sync_dist=True)

            if mIoU > self.best_mIoU.item():
                self.best_mIoU = mIoU
                self.metric.save(path=self.logger.log_dir)
            self.log_dict({"mIoU/best_mIoU": self.best_mIoU, "step": self.current_epoch}, on_epoch=True, logger=True,sync_dist=True)
            #self.log("mIoU/best_mIoU", self.best_mIoU, logger=True, sync_dist=True)

            dic_IoU = {}
            for id, iou in enumerate(IoU):
                #if self.config.DATASET.CLASS_LABELS is not []:
                if hasNotEmptyAttr(self.config.DATASET,"CLASS_LABELS"):
                    dic_IoU[str(id) + "-" + self.config.DATASET.CLASS_LABELS[id]] = "%.4f" % iou.item()
                else:
                    dic_IoU[id] = "%.4f" % iou.item()

            if self.global_rank == 0:
                log.info("EPOCH: %s", self.current_epoch)
                log.info("Best mIoU %.4f      Mean IoU: %.4f", self.best_mIoU, mIoU.item())
                log.info(dic_IoU)

    def test_step(self, batch, batch_idx):
        x, y_gt = batch
        y_pred = self(x)
        self.metric.update(y_gt.flatten(), y_pred["out"].argmax(1).flatten())

    def on_test_epoch_end(self) -> None:

        IoU, mIoU = self.metric.compute()

        dic_IoU = {}
        for id, iou in enumerate(IoU):
            if hasNotEmptyAttr(self.config.DATASET,"CLASS_LABELS"):
                dic_IoU[str(id) + "-" + self.config.DATASET.CLASS_LABELS[id]] = "%.4f" % iou.item()
            else:
                dic_IoU[id] = "%.4f" % iou.item()
        print(dic_IoU)
        print("Mean IoU:","%.4f" % mIoU.item())




@hydra.main(config_path="config", config_name="baseline")
def training_loop(cfg: DictConfig):
    ### SEEDING IF GIVEN BY CONFIG ####
    if hasNotEmptyAttr(cfg, "seed"):
        seed_everything(cfg.seed, workers=True)

    ### IMPORTING CALLBACKS USING HYDRA ####
    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    ### USING TENSORBOARD LOGGER ####
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="", default_hp_metric=False)

    AVAIL_GPUS = torch.cuda.device_count()
    log.info('Available GPUs: %s - %s', AVAIL_GPUS, torch.cuda.get_device_name())
    cfg.num_gpus = AVAIL_GPUS

    ### DEFINING DATASET ####
    dataModule=hydra.utils.instantiate(cfg.datamodule,_recursive_=False)

    ### DEFINING MODEL AND LOAD CHECKPOINT IF WANTED####
    if hasNotEmptyAttr(cfg,"finetune_from"):
        # +finetune_from = /home/l727r/Desktop/Target_Folder/Cityscape/hrnet/epochs\=400/2021-12-25_22-04-38_environment\=cluster_epochs\=400/checkpoints/best_epoch\=393_mIoU\=0.8144.ckpt
        log.info("finetune from: %s", cfg.finetune_from)
        model = SegModel.load_from_checkpoint(cfg.finetune_from, config=cfg)
    else:
        model = SegModel(config=cfg)


    ### INITIALIAZING TRAINER ####
    trainer_args = getattr(cfg, "pl_trainer") if hasNotEmptyAttr(cfg, "pl_trainer") else {}
    trainer = Trainer(
        max_epochs=cfg.epochs,
        gpus=AVAIL_GPUS,

        callbacks=callbacks,
        logger=tb_logger,
        #enable_checkpointing=True if cfg else False,

        strategy='ddp' if AVAIL_GPUS > 1 else None,
        #strategy=DDPPlugin(find_unused_parameters=False) if AVAIL_GPUS > 1 else None,

        **trainer_args
    )

    ### START TRAINING ####
    trainer.fit(model, dataModule)


def validation(ckpt_path,hp_path):

    hydra.initialize(config_path=hp_path)
    #cfg = hydra.compose(config_name="config",overrides=["MODEL.ADAPTED_PRETRAINED_WEIGHTS=""", "MODEL.PRETRAINED=False","MODEL.MSCALE_TRAINING=true","DATASET.ROOT=/home/l727r/Desktop/Cityscape"])
    cfg = hydra.compose(config_name="config",overrides=["MODEL.ADAPTED_PRETRAINED_WEIGHTS=""", "MODEL.PRETRAINED=False","DATASET.ROOT=/home/l727r/Desktop/Cityscape"])
    cfg.val_batch_size=1
    model = SegModel.load_from_checkpoint(ckpt_path, config=cfg)
    #print(cfg.dataset)
    #dataModule = getattr(DataModules, cfg.DATASET.NAME)(config=cfg)
    #try:
    dataModule = hydra.utils.instantiate(cfg.datamodule,_recursive_=False)
    #except:
    #dataModule = getattr(DataModules, "BaseDataModule")(config=cfg)



    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        precision= 16,
        benchmark= True
    )

    trainer.test(model,dataModule)


if __name__ == "__main__":
    training_loop()
    #path = "/home/l727r/Desktop/Target_Folder/Cityscape/hrnet/epochs\=400/2021-12-25_22-04-38_environment\=cluster_epochs\=400/checkpoints/best_epoch=393_mIoU\=0.8144.ckpt"
    #checkpoint = torch.load(path)
    #print(checkpoint.keys())
    #print(checkpoint["state_dict"])

    '''base="/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/"
    folder="MODEL.MSCALE_TRAINING=false_epochs=500"
    #folder="MODEL.MSCALE_TRAINING=false_epochs=400_lossfunction=wRMI"
    x=os.listdir(base+folder)
    #x=x[1:]
    #print(x)
    for name in x:
        #GlobalHydra.instance().clear()
        #hydra._internal.hydra.GlobalHydra.get_state().clear()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        name=os.path.join(folder,name)
        print(name)
        print("/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/"+name+"/checkpoints/best_epoch*.ckpt")

        ckpt_path=glob.glob("/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/"+name+"/checkpoints/best_epoch*.ckpt")
        print(ckpt_path)
        hp_path="../Target_Folder/Cityscape/hrnet_ocr_ms/MODEL.MSCALE_TRAINING=False_epochs=400/2022-01-05_12-06-13/hydra"
        #hp_path="../Target_Folder/Cityscape/hrnet_ocr_ms/"+name+"/hydra"
        validation(ckpt_path[-1],hp_path)

    #name="MODEL.MSCALE_TRAINING=False_epochs=400/2022-01-05_12-06-13"
    #"/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/MODEL.MSCALE_TRAINING=false_epochs=300/2022-01-12_07-50-38"
    #name="MODEL.MSCALE_TRAINING=false_epochs=300/2022-01-13_08-48-59"
    #p="MODEL.MSCALE_TRAINING=false_epochs=250"

    #ckpt_path=glob.glob("/home/l727r/Desktop/Target_Folder/Cityscape/hrnet_ocr_ms/"+name+"/checkpoints/best_epoch*.ckpt")
    #print(ckpt_path)
    #hp_path="../Target_Folder/Cityscape/hrnet_ocr_ms/MODEL.MSCALE_TRAINING=False_epochs=400/2022-01-05_12-06-13/hydra"
    #hp_path="../Target_Folder/Cityscape/hrnet_ocr_ms/"+name+"/hydra"
    #validation(ckpt_path[-1],hp_path)
    '''
