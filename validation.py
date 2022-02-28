import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import argparse
import os
import glob

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from Segmentation_Model import SegModel
from utils.utils import hasTrueAttr, hasNotEmptyAttr

from utils.utils import get_logger
log = get_logger(__name__)

from utils.visualization import show_data

# Inference Time
# https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f

def parse_key_into_cfg(cfg,key,value):
    if "." in key:  #Need for recursion
        key_a,key_b=key.split('.', 1)
        parse_key_into_cfg(cfg[key_a],key_b,value)
    else:
        try:
            cfg[key] = value
        except:
            print("Validation Override: key ", key, " not found")

def get_test_config(cfg):
    cfg.ORG_CWD = ""
    if hasNotEmptyAttr(cfg,"TESTING"):
        if hasNotEmptyAttr(cfg.TESTING,"OVERRIDES"):
            keys=cfg.TESTING.OVERRIDES.keys()
            for key in keys:
                #needed recursice Funciton to catch definiition like cfg.xx.xx ...
                parse_key_into_cfg(cfg,key,cfg.TESTING.OVERRIDES[key])

    return cfg

def validation(ckpt_dir,hydra_args,init=True):
    hydra.initialize(config_path="config")

    os.chdir(ckpt_dir)

    ###  load parameters from the checkpoint directory which are overritten ###
    overrides = OmegaConf.load(os.path.join("hydra","overrides.yaml"))
    train_overrides=["MODEL.PRETRAINED=False"]#,"pl_trainer.gpus=-1"]

    ### load local config and first override by the the parameters frm the checkpoint dir
    ### afterwards override the parameters from the commandline ###
    cfg = hydra.compose(config_name="baseline", overrides=overrides+hydra_args+train_overrides)

    ### change some testing specific parameters ###
    cfg=get_test_config(cfg)
    #print(cfg)
    ### load checkpoint and load model ###
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]
    log.info("Checkpoint Directory: %s",ckpt_file)
    model = SegModel.load_from_checkpoint(ckpt_file, strict=False,config=cfg)

    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="",
                                             sub_dir="validation")  # ,default_hp_metric=False)

    trainer_args = getattr(cfg, "pl_trainer") if hasNotEmptyAttr(cfg, "pl_trainer") else {}
    trainer = Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        **trainer_args
    )

    trainer.test(model, dataModule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--valdir', type=str, default="")
    args, hydra_args = parser.parse_known_args()

    validation(args.valdir,hydra_args)

