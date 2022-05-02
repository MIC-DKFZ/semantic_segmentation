from omegaconf import OmegaConf, DictConfig

import os
import glob


from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from Segmentation_Model import SegModel
from utils.utils import has_not_empty_attr, has_true_attr, log_hyperparameters

from utils.utils import get_logger

import hydra

log = get_logger(__name__)


@hydra.main(config_path="config", config_name="testing")
def validation(cfg: DictConfig) -> None:
    """
    Running the Testing/Validation
    Change Working Directory to ckpt_dir
    Load the hydra overrides from the ckpt_dir
    Compose config from config/testing.yaml with overwrites from the checkpoint and the overwrites
    from commandline
    Load Model, Datamodule, Logger and Trainer
    run testing

    Parameters
    ----------
    cfg : DictConfig
    """
    # save overrides from current run
    overrides_cl = hydra.core.hydra_config.HydraConfig.get().overrides.task
    # load overrides from the experiment in the checkpoint dir
    overrides_ckpt = OmegaConf.load(os.path.join("hydra", "overrides.yaml"))
    # compose config by override with overrides_ckpt, afterwards override with overrides_cl
    cfg = hydra.compose(config_name="testing", overrides=overrides_ckpt + overrides_cl)

    # load the best checkpoint and load the model
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]
    log.info("Checkpoint Directory: %s", ckpt_file)
    model = SegModel.load_from_checkpoint(ckpt_file, config=cfg, strict=False)

    # load the datamodule
    dataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    ### instantiate callbacks ###
    callbacks = []
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="validation",
        name="",
        version="",
        default_hp_metric=False,
    )

    # parsing the pl_trainer args and instantiate the trainer
    trainer_args = getattr(cfg, "pl_trainer") if has_not_empty_attr(cfg, "pl_trainer") else {}
    trainer = Trainer(callbacks=callbacks, logger=tb_logger, **trainer_args)

    # log hyperparameters
    log_hyperparameters(cfg, model, trainer)

    # run testing/validation
    trainer.test(model, dataModule)


if __name__ == "__main__":

    validation()
