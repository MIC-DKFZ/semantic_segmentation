import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from typing import Any
from tqdm import tqdm

import torch
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Initializes multi-GPU-friendly python command line logger
    Taken from:
    https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/__init__.py

    Parameters
    ----------
    name: str

    Returns
    -------
    logging.Logger :
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
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(
    #     ColoredFormatter(
    #         "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s]-"
    #         " %(message)s",
    #         log_colors={
    #             "DEBUG": "purple",
    #             "INFO": "green",
    #             "WARNING": "yellow",
    #             "ERROR": "red",
    #             "CRITICAL": "red",
    #         },
    #     )
    # )
    # print(ColoredFormatter)
    # logger.addHandler(console_handler)
    return logger


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """
    Controls which config parts are saved by Lightning loggers, additionally update hparams.yaml
    Taken and adopted from:
    https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/__init__.py

    Parameters
    ----------
    config : DictConfig
    model : pl.LightningModule
    trainer: pl.Trainer
    """
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config.MODEL.NAME
    hparams["dataset"] = config.DATASET.NAME
    hparams["metric"] = model.metric_name

    avail_GPUS = torch.cuda.device_count()
    selected_GPUS = config.pl_trainer.devices
    hparams["num_gpus"] = int(num_gpus(avail_GPUS, selected_GPUS))

    hparams["lossfunction"] = config.lossfunction
    hparams["optimizer"] = ""
    hparams["lr_scheduler"] = ""

    if hydra.core.hydra_config.HydraConfig.initialized():
        cfg = hydra.core.hydra_config.HydraConfig.get()
        if has_not_empty_attr(cfg.runtime.choices, "optimizer"):
            hparams["optimizer"] = cfg.runtime.choices.optimizer
        if has_not_empty_attr(cfg.runtime.choices, "lr_scheduler"):
            hparams["lr_scheduler"] = cfg.runtime.choices.lr_scheduler

    hparams["lr"] = config.lr
    hparams["epochs"] = config.epochs
    hparams["batch_size"] = config.batch_size
    hparams["precision"] = trainer.precision

    # save number of model parameters
    hparams["Parameter"] = sum(p.numel() for p in model.parameters())
    hparams["trainable Parameter"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metric = {
        "metric/best_" + model.metric_name: torch.nan,
        "Time/mTrainTime": torch.nan,
        "Time/mValTime": torch.nan,
    }
    # print(hparams, metric)
    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams, metric)

    # save resolved config in hparams.yaml
    OmegaConf.save(
        config=config,
        resolve=True,
        f=os.path.join(trainer.logger.log_dir, "hparams.yaml"),
    )


def num_gpus(avail_GPUS: int, selected_GPUS: Any) -> int:
    """
    Translating the num_gpus of pytorch lightning trainers into a raw number of used gpus
    Needed since lightning enables to pass gpu as int, list or string

    Parameters
    ----------
    avail_GPUS : int
        how many gpus are available
    selected_GPUS : Any
        num_gpus input argument for the pytorch lightning trainers

    Returns
    -------
    int :
        the number of used gpus
    """
    if selected_GPUS in [-1, "-1"]:
        num_gpus = avail_GPUS
    elif selected_GPUS in [0, "0", None]:
        num_gpus = 0
    elif isinstance(selected_GPUS, int):
        num_gpus = selected_GPUS
    elif isinstance(selected_GPUS, list):
        num_gpus = len(selected_GPUS)
    elif isinstance(selected_GPUS, str):
        num_gpus = len(selected_GPUS.split(","))
    return num_gpus


def first_from_dict(dictionary):
    return list(dictionary.values())[0]


def has_true_attr(obj: Any, attr: str) -> bool:
    """
    return True if obj contains attr and attr is true, else returns False

    Parameters
    ----------
    obj : Any
    attr : str

    Returns
    -------
    bool :
    """
    if hasattr(obj, attr):
        if obj[attr]:
            return True
    return False


def has_not_empty_attr(obj: Any, attr: str) -> bool:
    """
    return True if obj contains attr and attr is not empty, else returns False

    Parameters
    ----------
    obj : Any
    attr : str

    Returns
    -------
    bool :
    """
    if hasattr(obj, attr):
        if obj[attr] != None:
            return True
    return False


"""
def has_not_empty_attr_rec(obj: Any, attr: str) -> bool:
    # checking if the config contains a attribute and if this attribute is not empty
    split = attr.split(".", 1)
    key = split[0]
    attr = split[1:]
    if hasattr(obj, key):
        if attr == []:
            if obj[key] != None:
                return True
        else:
            return has_not_empty_attr_rec(obj[key], attr[0])
    return False
"""
