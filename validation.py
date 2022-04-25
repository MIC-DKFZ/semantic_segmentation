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

import hydra
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

def get_test_config(cfg,hydra_args):
    #hydra args are needed since all arguments should be overwritten from from commandline
    # otherwise the TESTING.OVERRIDES arguments could be overwritten
    alrady_override=[h.split("=")[0]for h in hydra_args]
    cfg.ORG_CWD = ""
    if hasNotEmptyAttr(cfg,"TESTING"):
        if hasNotEmptyAttr(cfg.TESTING,"OVERRIDES"):
            keys=cfg.TESTING.OVERRIDES.keys()
            for key in keys:
                if key not in alrady_override:
                    #needed recursice Funciton to catch definiition like cfg.xx.xx ...
                    parse_key_into_cfg(cfg,key,cfg.TESTING.OVERRIDES[key])

    return cfg

def validation(ckpt_dir,hydra_args,init=True):
    log = get_logger(__name__)
    hydra.initialize(config_path="config")#,strict=False)

    os.chdir(ckpt_dir)

    ###  load parameters from the checkpoint directory which are overritten ###
    overrides = OmegaConf.load(os.path.join("hydra","overrides.yaml"))
    #train_overrides=
    train_overrides=["MODEL.PRETRAINED=False"]#,"pl_trainer.gpus=-1"]

    ### load local config and first override by the the parameters frm the checkpoint dir
    ### afterwards override the parameters from the commandline ###
    #print(overrides)
    cfg = hydra.compose(config_name="baseline", overrides=overrides+hydra_args+train_overrides)#,strict=False)

    ### change some testing specific parameters ###
    cfg=get_test_config(cfg,hydra_args)
    #print(cfg)
    ### load checkpoint and load model ###
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]
    log.info("Checkpoint Directory: %s",ckpt_file)
    model = SegModel.load_from_checkpoint(ckpt_file, strict=False,config=cfg)

    ### load datamodule ###
    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    ### instantiate callbacks ###
    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    ### Sometimes Tensorboard doesnt create the 'validation' folder, to prevent this it is created manually ###
    if not os.path.exists('validation'):
        os.makedirs('validation')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="",
                                             sub_dir="validation")#,default_hp_metric=False)

    ### parsing the pl_trainer args ###
    trainer_args = getattr(cfg, "pl_trainer") if hasNotEmptyAttr(cfg, "pl_trainer") else {}
    trainer = Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        **trainer_args
    )

    ### run testing ###
    trainer.test(model, dataModule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #F:\Desktop\Target_Folder\Cityscapes\hrnet_ocr_ms\baseline__MODEL.MSCALE_TRAINING_False__epochs_65__lr_0.001\2022-01-25_11-18-48
    parser.add_argument('--ckpt_dir', type=str, default="")
    args, hydra_args = parser.parse_known_args()
    validation(args.ckpt_dir,hydra_args)
