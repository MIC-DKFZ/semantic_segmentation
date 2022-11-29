import glob

import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
import hydra
import os
import torch
from src.utils import get_logger

log = get_logger(__name__)


class Ensemble(nn.Module):
    def __init__(self, ckpts):
        super(Ensemble, self).__init__()
        models = []
        for ckpt in ckpts:

            # Init Model
            model_ckpt = OmegaConf.load(os.path.join(ckpt, "hparams.yaml")).model
            if hasattr(model_ckpt.cfg.MODEL, "PRETRAINED"):
                model_ckpt.cfg.MODEL.PRETRAINED = False
            model = hydra.utils.instantiate(model_ckpt)

            # Load State Dict
            ckpt_file = glob.glob(os.path.join(ckpt, "checkpoints", "best_*.ckpt"))[0]
            state_dict_ckpt = torch.load(ckpt_file, map_location={"cuda:0": "cpu"})
            if "state_dict" in state_dict_ckpt.keys():
                state_dict_ckpt = state_dict_ckpt["state_dict"]

            state_dict_ckpt = {
                k.replace("model.", "").replace("module.", ""): v
                for k, v in state_dict_ckpt.items()
            }
            model.load_state_dict(state_dict_ckpt)
            model.eval().cuda()
            models.append(model)

            log.info("{} loaded from ckpt {}".format(model_ckpt.cfg.MODEL.NAME, ckpt_file))

        self.models = models

    def forward(self, x):
        out = None
        for m in self.models:
            if out is None:
                out = m(x)["out"]
            else:
                out += m(x)["out"]

        out_avg = out / len(self.models)

        return out_avg
