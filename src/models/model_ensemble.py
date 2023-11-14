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
            model_ckpt = OmegaConf.load(join(ckpt, "hparams.yaml")).model.model

            model = hydra.utils.instantiate(model_ckpt)

            ckpt_file = glob.glob(join(ckpt, "checkpoints", ckpt_type + "_*.ckpt"))[0]
            ckpt_temp = update_state_dict(
                ckpt_file,
                model.state_dict(),
                ["model.", "module.", "_orig_mod."],
                f"Model {i}",
            )

            model.load_state_dict(ckpt_temp)

            model.eval().cuda()
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
            pred = m(x)
            if out is None:
                out = first_from_dict(pred) if isinstance(pred, dict) else pred
            else:
                out += first_from_dict(pred) if isinstance(pred, dict) else pred

        out_avg = out / len(self.models)

        return {"out": out_avg}
