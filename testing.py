import logging

logging.basicConfig(level=logging.INFO)

import os
import glob
import hydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from Segmentation_Model import SegModel
from utils.utils import has_not_empty_attr, has_true_attr, log_hyperparameters, get_logger


log = get_logger(__name__)


@hydra.main(config_path="config", config_name="testing")
def testing(cfg: DictConfig) -> None:
    """
    Running the Testing/Validation
    Using the ckpt_dir as Working Directory
    Load the hydra overrides from the ckpt_dir
    Compose config from config/testing.yaml with overwrites from the checkpoint and the overwrites
    from commandline
    (Optional) include Overrides defined in the config (TRAINING.OVERRIDES)
    Load Model, Datamodule, Logger and Trainer
    Run testing

    Parameters
    ----------
    cfg : DictConfig
        cfg given by hydra - build from config/testing.yaml + commandline arguments
    """
    # save overrides from the commandline for the current run
    overrides_cl = hydra.core.hydra_config.HydraConfig.get().overrides.task
    # load overrides from the experiment in the checkpoint dir
    overrides_ckpt = OmegaConf.load(os.path.join("hydra", "overrides.yaml"))
    # compose config by override with overrides_ckpt, afterwards override with overrides_cl
    cfg = hydra.compose(config_name="testing", overrides=overrides_ckpt + overrides_cl)

    # Get the TESTING.OVVERRIDES to check if additional parameters should be changed
    if has_not_empty_attr(cfg, "TESTING"):
        if has_not_empty_attr(cfg.TESTING, "OVERRIDES"):
            overrides_test = cfg.TESTING.OVERRIDES
            # Compose config again with including the new overrides
            cfg = hydra.compose(
                config_name="testing", overrides=overrides_ckpt + overrides_test + overrides_cl
            )

    # load the best checkpoint and load the model
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]
    log.info("Checkpoint Directory: %s", ckpt_file)
    model = SegModel.load_from_checkpoint(ckpt_file, config=cfg, strict=False)

    # load the datamodule
    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    # instantiate callbacks
    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="validation", name="", version="", default_hp_metric=False,
    )

    # parsing the pl_trainer args and instantiate the trainer
    trainer_args = getattr(cfg, "pl_trainer") if has_not_empty_attr(cfg, "pl_trainer") else {}
    trainer = Trainer(callbacks=callbacks, logger=tb_logger, **trainer_args)

    # log hyperparameters
    log_hyperparameters(cfg, model, trainer)

    # run testing/validation
    trainer.test(model, dataModule)


if __name__ == "__main__":

    testing()
