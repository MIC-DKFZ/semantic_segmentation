import glob
from os.path import join

import torch
from omegaconf import OmegaConf
import hydra

import torch.nn as nn

from src.utils.utils import get_logger
from src.utils.config_utils import first_from_dict
from src.utils.model_utils import update_state_dict

log = get_logger(__name__)


class Ensemble(nn.Module):
    def __init__(self, ckpts: list, ckpt_type: str = "best") -> None:
        """
        Ensemble Model
        Initialize a model for each ckpt in ckpt

        Parameters
        ----------
        ckpts:
            list of checkpoints (.ckpt file)
        ckpt_type
            best or last
        """
        super(Ensemble, self).__init__()
        models = []

        for i, ckpt in enumerate(ckpts):

            # Init Model
            model_ckpt = OmegaConf.load(join(ckpt, "hparams.yaml")).model
            if hasattr(model_ckpt.cfg.MODEL, "PRETRAINED"):
                model_ckpt.cfg.MODEL.PRETRAINED = False
            model = hydra.utils.instantiate(model_ckpt)

            # Load State Dict
            # if ckpt_type == "best":
            #     ckpt_file = glob.glob(join(ckpt, "checkpoints", "best_*.ckpt"))[0]
            # elif ckpt_type == "last":
            ckpt_file = glob.glob(join(ckpt, "checkpoints", ckpt_type + "_*.ckpt"))[0]

            model.load_state_dict(
                update_state_dict(
                    ckpt_file, model.state_dict(), ["model.", "module.", "_orig_mod."], f"Model {i}"
                )
            )
            # model.eval().cuda()
            models.append(model)

        self.models = models
        log.info(f"Initialized Model Ensemble using {len(self.models)} models")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward the input to each model and average the results. Only the first output of the models
        is used and returned

        Parameters
        ----------
        x

        Returns
        -------

        """
        out = None
        for m in self.models:
            if out is None:
                out = first_from_dict(m(x))
            else:
                out += first_from_dict(m(x))

        out_avg = out / len(self.models)

        return {"out": out_avg}
