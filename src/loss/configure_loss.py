from typing import Tuple, Dict, List, Any

import torch
import torch.nn
from torch.nn.modules.loss import _Loss
from omegaconf import DictConfig
from src.loss.rmi import RMILoss

from src.loss.Dice_Loss import DiceLoss as DL_custom
from src.loss.DC_CE_Loss import DC_and_CE_loss, TopKLoss, DC_and_topk_loss
from src.loss.Fishinspector import (
    BCE_PL,
    BCE_PLv2,
    SoftDiceLoss_Multilabel_PL,
    BCEWithLogitsLossPartlyLabeled,
)
from src.loss.multi_label import SparseBCEWithLogitsLoss
from src.utils.config_utils import has_not_empty_attr
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss


def get_loss_function_from_cfg(name_lf: str, cfg: DictConfig, device: torch.device) -> _Loss:
    """
    Instantiate the Lossfunction identified by name_lf
    Therefore get the needed parameters from the config

    Parameters
    ----------
    name_lf: str
        string identifier of the wanted loss function
    cfg: DictConfig
        config
    device: torch.device
        device which the weights should be on

    Returns
    -------
    Lossfunction
    """
    num_classes = cfg.num_classes
    ignore_index = cfg.ignore_index
    if name_lf == "CE":
        # loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        lCE = torch.nn.CrossEntropyLoss()
        loss_function = lambda pred, gt: lCE(pred, gt.to(torch.float16))
    elif name_lf == "wCE":
        weights = torch.FloatTensor(cfg.class_weights).to(device)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)

    elif name_lf == "RMI":
        loss_function = RMILoss(num_classes=num_classes, ignore_index=ignore_index)
    elif name_lf == "wRMI":
        weights = torch.FloatTensor(cfg.class_weights).to(device)
        loss_function = RMILoss(
            num_classes=num_classes, ignore_index=ignore_index, class_weights=weights
        )

    elif name_lf == "BCE":
        loss_function = torch.nn.BCELoss()
    elif name_lf == "BCEwL":
        BCEwL = torch.nn.BCEWithLogitsLoss()
        loss_function = lambda pred, gt: BCEwL(pred, gt.to(torch.float16))
    elif name_lf == "sBCEwL":
        loss_function = SparseBCEWithLogitsLoss()
    elif name_lf == "BCE_PL":
        loss_function = BCEWithLogitsLossPartlyLabeled()
    elif name_lf == "DicePL":
        loss_function = SoftDiceLoss_Multilabel_PL(batch_dice=True, smooth=0.0)
        # loss_function = lambda pred, gt, mask: lf(pred.clone(), gt.clone(), mask)
    elif name_lf == "mlDC":
        loss_function = DiceLoss(mode="multilabel")
    elif name_lf == "mlJL":
        loss_function = JaccardLoss(mode="multilabel")
    elif name_lf == "DC":
        loss_function = DL_custom(mode="multiclass", ignore_index=ignore_index)
    elif name_lf == "mlDC2":
        loss_function = DL_custom(mode="multilabel", ignore_index=ignore_index)
    elif name_lf == "mlDCls":
        loss_function = DiceLoss(mode="multilabel", smooth=0.1)
    elif name_lf == "mlJLls":
        loss_function = JaccardLoss(mode="multilabel", smooth=0.1)
    elif name_lf == "mlDC2ls":
        loss_function = DL_custom(mode="multilabel", ignore_index=ignore_index, smooth=0.1)
    elif name_lf == "DC_CE":
        DC_and_CE = DC_and_CE_loss(
            {"batch_dice": True, "smooth": 0, "do_bg": False},
            {"ignore_index": ignore_index},
            ignore_label=ignore_index,
        )
        loss_function = lambda pred, gt: DC_and_CE(pred.clone(), gt[:, None].clone())

    elif name_lf == "TOPK":
        TopK = TopKLoss(ignore_index=ignore_index)
        loss_function = lambda pred, gt: TopK(pred.clone(), gt[:, None])
    elif name_lf == "DC_TOPK":
        DC_TopK = DC_and_topk_loss(
            {"batch_dice": True, "smooth": 0, "do_bg": False},
            {"ignore_index": ignore_index},
            ignore_label=ignore_index,
        )
        loss_function = lambda pred, gt: DC_TopK(pred.clone(), gt[:, None])
    else:
        raise NotImplementedError("No Lossfunction found for {}".format(name_lf))
    return loss_function
