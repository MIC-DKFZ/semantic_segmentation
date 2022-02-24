import logging
logging.basicConfig(level=logging.INFO)
import os
import hydra

import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything

from utils.utils import hasTrueAttr, hasNotEmptyAttr,get_logger, num_gpus
from omegaconf import DictConfig, OmegaConf
from Segmentation_Model import SegModel
from validation import validation

log = get_logger(__name__)

#OmegaConf resolver for preventing problems in the output path
OmegaConf.register_new_resolver('path_formatter', lambda s: s.replace("[","").replace("]","").replace(",","_").replace("=","_").replace("/",".").replace("+",""))
@hydra.main(config_path="config", config_name="baseline")
def training_loop(cfg: DictConfig):
    log.info("Output Directory: %s",os.getcwd())
    ### SEEDING IF GIVEN BY CONFIG ###
    if hasNotEmptyAttr(cfg, "seed"):
        seed_everything(cfg.seed, workers=True)

    ### IMPORTING CALLBACKS USING HYDRA ###
    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)
    ### Adding a Checkpoint Callback if checkpointing is enabled ###
    if hasTrueAttr(cfg.pl_trainer,"enable_checkpointing"):
        callbacks.append(hydra.utils.instantiate(cfg.ModelCheckpoint))

    ### USING TENSORBOARD LOGGER ####
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="", default_hp_metric=False)

    ### LOGGING INFOS ABOUT GPU SETTING ###
    avail_GPUS=torch.cuda.device_count()
    selected_GPUS=cfg.pl_trainer.gpus
    number_gpus=num_gpus(avail_GPUS,selected_GPUS)

    log.info('Available GPUs: %s - %s', avail_GPUS, torch.cuda.get_device_name())
    log.info('Number of used GPUs: %s    Selected GPUs: %s', number_gpus, cfg.pl_trainer.gpus)
    log.info('CUDA version: %s', torch._C._cuda_getCompiledVersion())


    ### DEFINING DATASET ####
    dataModule=hydra.utils.instantiate(cfg.datamodule,_recursive_=False)

    ### DEFINING MODEL AND LOAD CHECKPOINT IF WANTED ###
    ### cfg.finetune_from should be the path to a .ckpt file ###
    if hasNotEmptyAttr(cfg,"finetune_from"):
        log.info("finetune from: %s", cfg.finetune_from)
        model = SegModel.load_from_checkpoint(cfg.finetune_from, config=cfg)
    else:
        model = SegModel(config=cfg)

    ### INITIALIAZING TRAINER ###
    trainer_args = getattr(cfg, "pl_trainer") if hasNotEmptyAttr(cfg, "pl_trainer") else {}
    trainer = Trainer(
        callbacks=callbacks,
        logger=tb_logger,

        strategy='ddp' if number_gpus > 1 else None,

        **trainer_args
    )

    ### START TRAINING ###
    trainer.fit(model, dataModule)

    ### OPTIONAL TESTING, USED WHEN MODEL IS TESTED UNDER DIFFERENT CONDITIONS THAN TRAINING ###
    ### Currently this doesnt work for multi gpu training - some kind of cuda error
    #if hasattr(cfg, "TESTING"):
    #    if hasTrueAttr(cfg.TESTING,"TEST_AFTERWARDS"):
    #        torch.cuda.empty_cache()
    #        # Hydra environment has to be cleared since a seperate one is creted during validation
    #        hydra.core.global_hydra.GlobalHydra.instance().clear()
    #        validation(os.getcwd(),[],False)


if __name__ == "__main__":

    training_loop()
