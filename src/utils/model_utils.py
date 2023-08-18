import os
import torch
from src.utils.utils import get_logger

log = get_logger(__name__)


def update_state_dict(weight_file, model_dict, replace=None, name="Model"):

    if os.path.isfile(weight_file):
        # Loading weights from path and extract state_dict
        log.info(f"Loading {name}: Loading weights: {weight_file}")
        pretrained_dict = torch.load(weight_file, map_location={"cuda:0": "cpu"})
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = pretrained_dict["state_dict"]

        # Replace substrings in the pretrained dict if given
        if replace is not None:
            for r in replace:
                if isinstance(r, tuple):
                    pretrained_dict = {k.replace(r[0], r[1]): v for k, v in pretrained_dict.items()}
                else:
                    pretrained_dict = {k.replace(r, ""): v for k, v in pretrained_dict.items()}

        # Check if layer names match between model_dict and pretrained_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        # Log info about layers in the model_dict which are not found in the pretrained_dict
        no_match = set(model_dict) - set(pretrained_dict)
        if len(no_match):
            num = len(no_match)
            log_info = no_match if num < 5 else list(no_match)[:5] + ["..."]
            log.warning(f"Loading {name}: No weights found for {num} layers: {log_info}")

        # Check if layer shapes match between model_dict and pretrained_dict
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if v.shape == model_dict[k].shape
        }
        # Log info about layers with a shape mismatch between model_dict and pretrained_dict
        shape_mismatch = (set(model_dict) - set(pretrained_dict)) - set(no_match)
        if len(shape_mismatch):
            num = len(shape_mismatch)
            log_info = shape_mismatch if num < 5 else list(shape_mismatch)[:5] + ["..."]
            log.warning(f"Loading {name}: Shape Mismatch for {num} layers: {log_info}")

        # Update and return model_dict
        if pretrained_dict == {}:
            log.warning(f"Loading {name}: No Weights are updated")
        else:
            model_dict.update(pretrained_dict)
            log.info(
                f"Loading {name}: {len(pretrained_dict.keys())} of {len(model_dict.keys())} layers"
                " are updated"
            )
        return model_dict
    else:
        raise NotImplementedError(f"Loading {name}: No Pretrained Weights found for {weight_file}")
