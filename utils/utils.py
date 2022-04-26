import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import os

import torch
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO)


def get_logger(name=__name__) -> logging.Logger:
    """
    Taken from:
    https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/__init__.py
        Initializes multi-GPU-friendly python command line logger.
    """

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """
    Taken and slightly adopted from:
    https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/__init__.py
        Controls which config parts are saved by Lightning loggers.
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config.MODEL.NAME
    hparams["dataset"] = config.DATASET.NAME
    hparams["metric"] = model.metric_name

    avail_GPUS = torch.cuda.device_count()
    selected_GPUS = config.pl_trainer.gpus
    hparams["num_gpus"] = int(num_gpus(avail_GPUS, selected_GPUS))

    hparams["lossfunction"] = config.lossfunction
    hparams["optimizer"] = ""
    hparams["lr_scheduler"] = ""

    if hydra.core.hydra_config.HydraConfig.initialized():
        cfg = hydra.core.hydra_config.HydraConfig.get()
        if hasNotEmptyAttr(cfg.runtime.choices, "optimizer"):
            hparams["optimizer"] = cfg.runtime.choices.optimizer
        if hasNotEmptyAttr(cfg.runtime.choices, "lr_scheduler"):
            hparams["lr_scheduler"] = cfg.runtime.choices.lr_scheduler

    hparams["lr"] = config.lr
    hparams["epochs"] = config.epochs
    hparams["batch_size"] = config.batch_size
    hparams["precision"] = trainer.precision

    # save number of model parameters
    hparams["Parameter"] = sum(p.numel() for p in model.parameters())
    hparams["trainable Parameter"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metric = {"metric/best_" + model.metric_name: 0, "Time/mTrainTime": 0, "Time/mValTime": 0}

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams, metric)
    OmegaConf.save(
        config=config, resolve=True, f=os.path.join(trainer.logger.log_dir, "hparams.yaml")
    )


def num_gpus(avail_GPUS: int, selected_GPUS: int) -> int:
    # Transfering pytorch lightning gpu argument into the number of gpus
    # Needed since lightning enables to pass gpu as list or string
    if selected_GPUS in [-1, "-1"]:
        num_gpus = avail_GPUS
    elif selected_GPUS in [0, "0", None]:
        num_gpus = 0
    elif isinstance(selected_GPUS, int):
        num_gpus = selected_GPUS
    elif isinstance(selected_GPUS, list):
        num_gpus = len(selected_GPUS)
    elif isinstance(selected_GPUS, str):
        num_gpus = len(selected_GPUS.split((",")))
    return num_gpus


def hasTrueAttr(obj, attr: str) -> bool:
    # checking if the config contains a attribute and if this attribute is true
    if hasattr(obj, attr):
        if obj[attr]:
            return True
    return False


def hasNotEmptyAttr(obj, attr: str) -> bool:
    # checking if the config contains a attribute and if this attribute is not empty
    if hasattr(obj, attr):
        if obj[attr] != None:
            return True
    return False


def hasNotEmptyAttr_rec(obj, attr: str) -> bool:
    # checking if the config contains a attribute and if this attribute is not empty
    split = attr.split(".", 1)
    key = split[0]
    attr = split[1:]
    if hasattr(obj, key):
        if attr == []:
            if obj[key] != None:
                return True
        else:
            return hasNotEmptyAttr_rec(obj[key], attr[0])
    return False
