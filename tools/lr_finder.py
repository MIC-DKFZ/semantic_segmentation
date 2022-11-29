import argparse
import os
import logging
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
import hydra
import torch

from trainers.Semantic_Segmentation_Trainer import SegModel

from src.utils import has_true_attr, has_not_empty_attr, get_logger, num_gpus


log = get_logger(__name__)


def find_lr(overrides_cl: list, num_training_samples: int) -> None:
    """
    Implementation for using Pytorch Lightning learning rate finder

    Parameters
    ----------
    overrides_cl : list of strings
        arguments from the commandline to overwrite the config
    num_training_samples: int
        how many samples to use for lr finding
    """
    # initialize hydra
    hydra.initialize(config_path="../config", version_base="1.1")

    overrides_cl.append("ORG_CWD=./")
    cfg = hydra.compose(config_name="baseline", overrides=overrides_cl)

    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)
    # Adding a Checkpoint Callback if checkpointing is enabled
    if has_true_attr(cfg.pl_trainer, "enable_checkpointing"):
        cfg.pl_trainer.enable_checkpointing = False

    # Using tensorboard logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=".", name="", version="", default_hp_metric=False
    )

    # Logging information about gpu setup
    avail_GPUS = torch.cuda.device_count()
    selected_GPUS = cfg.pl_trainer.devices
    number_gpus = num_gpus(avail_GPUS, selected_GPUS)

    log.info("Available GPUs: %s - %s", avail_GPUS, torch.cuda.get_device_name())
    log.info("Number of used GPUs: %s    Selected GPUs: %s", number_gpus, cfg.pl_trainer.devices)
    log.info("CUDA version: %s", torch._C._cuda_getCompiledVersion())

    # Defining the datamodule
    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    # Defining model and load checkpoint if wanted
    # cfg.finetune_from should be the path to a .ckpt file
    # if has_not_empty_attr(cfg, "finetune_from"):
    #     log.info("finetune from: %s", cfg.finetune_from)
    #     model = SegModel.load_from_checkpoint(cfg.finetune_from, strict=False, config=cfg)
    # else:
    #     model = SegModel(config=cfg)
    if hasattr(cfg, "num_example_predictions"):
        cfg.num_example_predictions = 0
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

    # ddp=DDPPlugin(find_unused_parameters=False) if number_gpus > 1 else None
    trainer = Trainer(
        callbacks=callbacks,
        logger=tb_logger,
        strategy="ddp" if number_gpus > 1 else None,
        sync_batchnorm=True if number_gpus > 1 else False,
        auto_lr_find="config.lr",
        **trainer_args,
    )
    lr_finder = trainer.tuner.lr_find(
        model=model,
        datamodule=dataModule,
        num_training=num_training_samples,
    )
    print("lr suggestion: ", lr_finder.suggestion())

    fig = lr_finder.plot(suggest=True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_training_samples",
        type=int,
        default=100,
        help="how many samples to use for lr finding",
    )
    args, overrides = parser.parse_known_args()
    num_training_samples = args.num_training_samples
    find_lr(overrides, num_training_samples)
