import hydra
from omegaconf import OmegaConf
import os
from typing import Any
import numpy as np
import glob
from os.path import join


def first_from_dict(dictionary: dict) -> Any:
    """
    Return first Item from Dict
    """
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


def get_CV_ensemble_config(ckpt_dir, ckpt_type="best"):
    folds = np.unique([f for f in os.listdir(ckpt_dir) if "fold_" in f and "fold_all" not in f])
    ckpts = []
    for fold in folds:
        ckpt = glob.glob(join(ckpt_dir, fold, "*"))
        if ckpt != []:
            ckpts.append(ckpt[-1])
    return {"_target_": "models.model_ensemble.Ensemble", "ckpts": ckpts, "ckpt_type": ckpt_type}


def build_predict_config(file, overrides_cl):
    # Load overrides from the experiment in the checkpoint dir
    overrides_ckpt = OmegaConf.load(file)
    # Compose config by override with overrides_ckpt, afterwards override with overrides_cl
    cfg = hydra.compose(config_name="testing", overrides=overrides_ckpt + overrides_cl)

    # Get the TESTING.OVVERRIDES to check if additional parameters should be changed
    if has_not_empty_attr(cfg, "TESTING"):
        if has_not_empty_attr(cfg.TESTING, "OVERRIDES"):
            overrides_test = cfg.TESTING.OVERRIDES
            # Compose config again with including the new overrides
            cfg = hydra.compose(
                config_name="testing",
                overrides=overrides_ckpt + overrides_test + overrides_cl,
            )
    # Causes since hydra.runtime parameters are not initialized using hydra.initialize
    cfg.logging.output_dir = ""
    # cfg.OUTPUT_DIR = ""
    return cfg


def build_test_config(file):
    # Save overrides from the commandline for the current run
    overrides_cl = hydra.core.hydra_config.HydraConfig.get().overrides.task
    # Load overrides from the experiment in the checkpoint dir
    overrides_ckpt = OmegaConf.load(file)
    # Compose config by override with overrides_ckpt, afterwards override with overrides_cl
    cfg = hydra.compose(config_name="testing", overrides=overrides_ckpt + overrides_cl)
    # Get the TESTING.OVVERRIDES to check if additional parameters should be changed
    if has_not_empty_attr(cfg, "TESTING"):
        if has_not_empty_attr(cfg.TESTING, "OVERRIDES"):
            overrides_test = cfg.TESTING.OVERRIDES
            # Compose config again with including the new overrides
            cfg = hydra.compose(
                config_name="testing",
                overrides=overrides_ckpt + overrides_test + overrides_cl,
            )
    return cfg
