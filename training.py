from typing import List
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)

import hydra
from omegaconf import DictConfig
import torch
import lightning as L
from lightning import LightningDataModule, LightningModule, Callback, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies.ddp import DDPStrategy

from src.utils.config_utils import has_true_attr, has_not_empty_attr
from src.utils.utils import (
    get_logger,
    num_gpus,
    log_hyperparameters,
    register_resolvers,
    set_lightning_logging,
)

log = get_logger(__name__)
set_lightning_logging()
register_resolvers()

# @hydra.main(config_path="config", config_name="training", version_base="1.3")
@hydra.main(config_path="config", config_name="training", version_base="1.3")
def training(cfg: DictConfig) -> None:
    """
    Instantiating Callbacks, Logger, Datamodule, Model and Trainer
    Run the lightning training loop
    Test the trained model afterward

    Parameters
    ----------
    cfg : DictConfig
        cfg given by hydra - build from config/training.yaml + commandline arguments
    """
    # Logging - path to Output
    log.info("Output Directory: %s", os.getcwd())

    # Logging - Information about GPU-setup
    avail_GPUS = torch.cuda.device_count()
    selected_GPUS = cfg.pl_trainer.devices
    number_gpus = num_gpus(avail_GPUS, selected_GPUS)
    log.info(f"Available GPUs: {avail_GPUS} - {torch.cuda.get_device_name()}")
    log.info(f"Number of used GPUs: {number_gpus}    Selected GPUs: {cfg.pl_trainer.devices}")
    log.info(f"CUDA version: {torch._C._cuda_getCompiledVersion()}")
    log.info(f"Pytorch version: {torch.__version__}")

    # Seeding - Unfortunately no True deterministic since deterministic=True causes errors with CE
    if has_not_empty_attr(cfg, "seed"):
        log.info(f"Seed everything with seed: {cfg.seed}")
        L.seed_everything(cfg.seed, workers=True)
        # cfg.pl_trainer.deterministic = True
        # cfg.pl_trainer.benchmark = False

    # Callbacks - Instantiating all Callbacks in List
    callbacks: List[Callback] = []
    for _, cb_conf in cfg.callbacks.items():
        if cb_conf is not None:
            cb: Callback = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    # Checkpointing - Adding and Instantiating Callback if checkpointing is enabled
    if has_true_attr(cfg.pl_trainer, "enable_checkpointing"):
        callbacks.append(hydra.utils.instantiate(cfg.ModelCheckpoint))

    # Logger - Instantiating
    logger: Logger = hydra.utils.instantiate(cfg.logger)

    # Datamodule - Instantiating
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    # Model - Instantiating and load weights from a Checkpoint if given
    if has_not_empty_attr(cfg, "finetune_from"):
        log.info("finetune from: %s", cfg.finetune_from)  # should be path to a .ckpt file
        cfg.trainermodule._target_ += ".load_from_checkpoint"
        model: LightningModule = hydra.utils.instantiate(
            cfg.trainermodule, cfg.finetune_from, strict=False, cfg=cfg, _recursive_=False
        )
    else:
        model: LightningModule = hydra.utils.instantiate(cfg.trainermodule, cfg, _recursive_=False)

    # Trainer - Instantiating with trainer_args from config (cfg.pl_trainer)
    trainer_args: dict = getattr(cfg, "pl_trainer") if has_not_empty_attr(cfg, "pl_trainer") else {}
    trainer: Trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False) if number_gpus > 1 else "auto",
        sync_batchnorm=True if number_gpus > 1 else False,
        **trainer_args,
    )

    # Logging - Hyperparameters, if-statement is needed to catch fast_dev_run
    if not has_true_attr(cfg.pl_trainer, "fast_dev_run"):
        log_hyperparameters(cfg, model, trainer)

    # Model - Compile
    if has_true_attr(cfg, "compile"):
        model = torch.compile(model)

    # Training
    trainer.fit(
        model,
        datamodule,
        ckpt_path=cfg.continue_from if hasattr(cfg, "continue_from") else None,
    )

    # Testing
    if has_true_attr(cfg.pl_trainer, "enable_checkpointing"):
        ckpt_path = "best"
    else:
        ckpt_path = None
    log.info(f"Start Testing with {ckpt_path} checkpoint")
    trainer.test(model, datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    training()
