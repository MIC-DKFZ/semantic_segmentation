import logging

logging.basicConfig(level=logging.INFO)

import os
import hydra
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

from src.utils import (
    has_true_attr,
    has_not_empty_attr,
    get_logger,
    num_gpus,
    log_hyperparameters,
)
from omegaconf import DictConfig, OmegaConf

# from trainers.Semantic_Segmentation_Trainer import SegModel
# from trainers.Instance_Segmentation_Trainer import InstSegModel as SegModel

log = get_logger(__name__)


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
    .replace("+", ""),
)


@hydra.main(config_path="config", config_name="training", version_base="1.3")
def training_loop(cfg: DictConfig):
    """
    Running Training
    import Callbacks and initialize Logger
    Load Model, Datamodule and Trainer
    Train the model

    Parameters
    ----------
    cfg :
        cfg given by hydra - build from config/training.yaml + commandline argumentss
    """
    # for k, v in logging.Logger.manager.loggerDict.items():
    #     if not isinstance(v, logging.PlaceHolder):
    #         print(k, v.handlers)
    log.info("Output Directory: %s", os.getcwd())
    # Seeding if given by config
    if has_not_empty_attr(cfg, "seed"):
        seed_everything(cfg.seed, workers=True)

    # Importing callbacks using hydra
    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)
    # Adding a Checkpoint Callback if checkpointing is enabled
    if has_true_attr(cfg.pl_trainer, "enable_checkpointing"):
        callbacks.append(hydra.utils.instantiate(cfg.ModelCheckpoint))

    # Using tensorboard logger
    # tb_logger = pl_loggers.TensorBoardLogger(
    #     save_dir=".", name="", version="", default_hp_metric=False
    # )
    tb_logger = hydra.utils.instantiate(cfg.logger)

    # Logging information about gpu setup
    avail_GPUS = torch.cuda.device_count()
    selected_GPUS = cfg.pl_trainer.devices
    number_gpus = num_gpus(avail_GPUS, selected_GPUS)
    log.info("Available GPUs: %s - %s", avail_GPUS, torch.cuda.get_device_name())
    log.info(
        "Number of used GPUs: %s    Selected GPUs: %s",
        number_gpus,
        cfg.pl_trainer.devices,
    )
    log.info(
        "CUDA version: {}    Pytorch version: {}".format(
            torch._C._cuda_getCompiledVersion(), torch.__version__
        )
    )

    # Defining the datamodule
    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    # Defining model and load checkpoint if wanted
    # cfg.finetune_from should be the path to a .ckpt file
    if has_not_empty_attr(cfg, "finetune_from"):
        log.info("finetune from: %s", cfg.finetune_from)
        cfg.trainermodule._target_ += ".load_from_checkpoint"
        model = hydra.utils.call(
            cfg.trainermodule, cfg.finetune_from, strict=False, model_config=cfg, _recursive_=False
        )
        # model = SegModel.load_from_checkpoint(cfg.finetune_from, strict=False, config=cfg)
    else:
        # model = SegModel(config=cfg)
        model = hydra.utils.instantiate(cfg.trainermodule, cfg, _recursive_=False)

    # Initializing trainers
    trainer_args = getattr(cfg, "pl_trainer") if has_not_empty_attr(cfg, "pl_trainer") else {}
    ddp = DDPStrategy(find_unused_parameters=False)  # if number_gpus > 1 else None
    trainer = Trainer(
        callbacks=callbacks,
        logger=tb_logger,
        strategy=ddp if number_gpus > 1 else "auto",
        # strategy="ddp_find_unused_parameters_false" if number_gpus > 1 else None,
        sync_batchnorm=True if number_gpus > 1 else False,
        **trainer_args
    )

    # Log experiment, if-statement is needed to catch fast_dev_run
    if not has_true_attr(cfg.pl_trainer, "fast_dev_run"):
        log_hyperparameters(cfg, model, trainer)

    if has_true_attr(cfg, "compile"):
        model.model = torch.compile(model.model)
    # Start training
    trainer.fit(
        model,
        dataModule,
        ckpt_path=cfg.continue_from if hasattr(cfg, "continue_from") else None,
    )


if __name__ == "__main__":
    training_loop()
