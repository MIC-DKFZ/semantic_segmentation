from typing import List
import os
from os.path import join
import sys
import glob
import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)

import hydra
from omegaconf import DictConfig
import lightning as L
from lightning import LightningDataModule, LightningModule, Callback, Trainer
from lightning.pytorch.loggers import Logger

from src.utils.config_utils import has_not_empty_attr, get_CV_ensemble_config, build_test_config
from src.utils.utils import get_logger, log_hyperparameters, set_lightning_logging

log = get_logger(__name__)
set_lightning_logging()


@hydra.main(config_path="config", config_name="testing", version_base="1.3")
def testing(cfg: DictConfig) -> None:
    """
    Compose config from config/testing.yaml with overwrites from the checkpoint and the overwrites
    from commandline
    Instantiating Callbacks, Logger, Datamodule, Model and Trainer
    Enables to test a CV ensemble as well as a single model dependent on the cfg.ckpt_file
    Run the lightning testing loop

    Parameters
    ----------
    cfg : DictConfig
        cfg given by hydra - build from config/testing.yaml + commandline arguments

    Requirements
    ----------
    cfg.ckpt_file: str
        a) path to a singleoutput folder
           e.g. .../logs/Cityscapes/hrnet_v1/run__batch_size_3/2023-08-18_10-13-05
        b) path to output folder of a cross validation, folder has to contain at least one fold_x
           e.g. .../logs/Cityscapes/hrnet_v1/run__batch_size_3/
    """
    # Config - Check if cfg.ckpt_dir points to a single model or a CV folder
    ensemble_CV: bool = any([x.startswith("fold_") for x in os.listdir(cfg.ckpt_dir)])

    # Config - Load the config from the Checkpoint
    if ensemble_CV:
        # All runs inside CV have the same config (except dataset.fold which is not relevant here)
        file: str = glob.glob(join(cfg.ckpt_dir, "fold_*", "*", "hydra", "overrides.yaml"))[0]
    else:
        file: str = join(cfg.ckpt_dir, ".hydra", "overrides.yaml")

    # Config - Compose the config
    cfg: DictConfig = build_test_config(file)

    # Model - Instantiating and load weights from a Checkpoint
    if ensemble_CV:
        cfg.model: DictConfig = get_CV_ensemble_config(cfg.ckpt_dir)
        model: LightningModule = hydra.utils.instantiate(
            cfg.trainermodule, model_config=cfg, _recursive_=False
        )
    else:
        ckpt_file: str = glob.glob(os.path.join(cfg.ckpt_dir, "checkpoints", "best_*"))[0]
        log.info("Checkpoint Directory: %s", ckpt_file)

        cfg.trainermodule._target_ += ".load_from_checkpoint"
        model: LightningModule = hydra.utils.instantiate(
            cfg.trainermodule,
            ckpt_file,
            strict=False,
            model_config=cfg,
            _recursive_=False,
        )

    # Datamodule - Instantiating
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    # Callbacks - Instantiating
    callbacks: List[Callback] = []
    for _, cb_conf in cfg.callbacks.items():
        if cb_conf is not None:
            cb: Callback = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    # Logger - Instantiating
    logger: Logger = hydra.utils.instantiate(cfg.logger)

    # Trainer - Instantiating with trainer_args from config (cfg.pl_trainer)
    trainer_args: dict = getattr(cfg, "pl_trainer") if has_not_empty_attr(cfg, "pl_trainer") else {}
    trainer: Trainer = L.Trainer(callbacks=callbacks, logger=logger, **trainer_args)

    # Logging - Hyperparameters
    log_hyperparameters(cfg, model, trainer)

    # Testing
    trainer.test(model, datamodule)


if __name__ == "__main__":
    testing()
