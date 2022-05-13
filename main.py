import logging

logging.basicConfig(level=logging.INFO)

import os
import hydra

import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin

from utils.utils import (
    has_true_attr,
    has_not_empty_attr,
    get_logger,
    num_gpus,
    log_hyperparameters,
)
from omegaconf import DictConfig, OmegaConf
from Segmentation_Model import SegModel

log = get_logger(__name__)


# OmegaConf resolver for preventing problems in the output path
OmegaConf.register_new_resolver(
    "path_formatter",
    lambda s: s.replace("[", "")
    .replace("]", "")
    .replace("}", "")
    .replace("{", "")
    .replace(")", "")
    .replace("(", "")
    .replace(",", "_")
    .replace("=", "_")
    .replace("/", ".")
    .replace("+", ""),
)


@hydra.main(config_path="config", config_name="baseline")
def training_loop(cfg: DictConfig):
    """
    Running Training
    import Callbacks and initialize Logger
    Load Model, Datamodule and Trainer
    Train the model

    Parameters
    ----------
    cfg :
        cfg given by hydra - build from config/baseline.yaml + commandline argumentss
    """
    log.info("Output Directory: %s", os.getcwd())
    # seeding if given by config
    if has_not_empty_attr(cfg, "seed"):
        seed_everything(cfg.seed, workers=True)

    # importing callbacks using hydra
    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)
    # Adding a Checkpoint Callback if checkpointing is enabled
    if has_true_attr(cfg.pl_trainer, "enable_checkpointing"):
        callbacks.append(hydra.utils.instantiate(cfg.ModelCheckpoint))

    # using tensorboard logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=".", name="", version="", default_hp_metric=False
    )

    # logging information about gpu setting
    avail_GPUS = torch.cuda.device_count()
    selected_GPUS = cfg.pl_trainer.gpus
    number_gpus = num_gpus(avail_GPUS, selected_GPUS)

    log.info("Available GPUs: %s - %s", avail_GPUS, torch.cuda.get_device_name())
    log.info("Number of used GPUs: %s    Selected GPUs: %s", number_gpus, cfg.pl_trainer.gpus)
    log.info("CUDA version: %s", torch._C._cuda_getCompiledVersion())

    # defining dataset
    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    # defining model and load checkpoint if wanted
    # cfg.finetune_from should be the path to a .ckpt file
    if has_not_empty_attr(cfg, "finetune_from"):
        log.info("finetune from: %s", cfg.finetune_from)
        model = SegModel.load_from_checkpoint(cfg.finetune_from, strict=False, config=cfg)
    else:
        model = SegModel(config=cfg)
    # quit()
    # initialiazing trainer
    trainer_args = getattr(cfg, "pl_trainer") if has_not_empty_attr(cfg, "pl_trainer") else {}

    # ddp=DDPPlugin(find_unused_parameters=False) if number_gpus > 1 else None
    trainer = Trainer(
        callbacks=callbacks,
        logger=tb_logger,
        strategy="ddp" if number_gpus > 1 else None,
        **trainer_args
    )

    # log hyperparameters, if-statement is needed to catch fast_dev_run
    if hasattr(trainer.logger, "log_dir"):
        log_hyperparameters(cfg, model, trainer)

    # start training
    trainer.fit(
        model,
        dataModule,
        ckpt_path=cfg.continue_from if hasattr(cfg, "continue_from") else None,
    )


if __name__ == "__main__":

    training_loop()
