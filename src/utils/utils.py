from typing import Any
import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.utilities import rank_zero_only

from src.utils.config_utils import has_true_attr, has_not_empty_attr


def set_lightning_logging():
    """
    remove Handlers from all lightning logging instances to prevent conflicts with hydra logging
    Preventing double printing and logging

    Returns
    -------

    """
    all_loggers = logging.Logger.manager.loggerDict

    # Filter out the loggers that are not instances of logging.Logger
    active_loggers = [
        logger for logger in all_loggers.values() if isinstance(logger, logging.Logger)
    ]

    # Check if logger is connected to lightning, if true remove the handler
    for logger in active_loggers:
        if logger.handlers != [] and "lightning" in logger.name:
            logger.removeHandler(logger.handlers[0])


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
    return logger


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: L.LightningModule,
    trainer: L.Trainer,
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
    hparams["model"] = config.model.name
    hparams["dataset"] = config.dataset.name
    hparams["metric"] = config.metric.name

    avail_GPUS = torch.cuda.device_count()
    selected_GPUS = config.pl_trainer.devices
    hparams["num_gpus"] = int(num_gpus(avail_GPUS, selected_GPUS))

    hparams["lossfunction"] = config.loss.function
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
        "metric/best_" + model.metric_cfg.name: torch.nan,
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
    Translating the num_gpus of pytorch lightning trainer into a raw number of used gpus
    Needed since lightning enables to pass gpu as int, list or string

    Parameters
    ----------
    avail_GPUS : int
        how many gpus are available
    selected_GPUS : Any
        num_gpus input argument for the pytorch lightning trainer

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


def register_resolvers():

    # OmegaConf.register_new_resolver(
    #     "fold_formatter", lambda s: f"fold_{s.fold}" if has_not_empty_attr(s, "fold") else ""
    # )
    OmegaConf.register_new_resolver(
        "fold_formatter", lambda s: f"fold_{s}" if s is not None else ""
    )
    # OmegaConf resolver for preventing problems in the output path
    # Removing all characters which can cause problems or are not wanted in a directory name
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
        .replace("+", "")
        .replace("@", "."),
    )
