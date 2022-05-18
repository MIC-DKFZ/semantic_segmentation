import torch
import torch.nn
from omegaconf import DictConfig
from utils.loss.rmi import RMILoss
from utils.loss.Dice_Loss import DiceLoss
from utils.loss.DC_CE_Loss import DC_and_CE_loss, TopKLoss, DC_and_topk_loss


def get_loss_function_from_cfg(name_lf: str, cfg: DictConfig) -> list:
    """
    Instantiate the Lossfunction identified by name_lf
    Therefore get the needed parameters from the config

    Parameters
    ----------
    name_lf: str
        string identifier of the wanted loss function
    cfg : DictConfig
        complete config

    Returns
    -------
    list of Lossfunctions
    """
    num_classes = cfg.DATASET.NUM_CLASSES
    ignore_index = cfg.DATASET.IGNORE_INDEX
    if name_lf == "CE":
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    elif name_lf == "wCE":
        weights = torch.FloatTensor(cfg.DATASET.CLASS_WEIGHTS).cuda()
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)

    elif name_lf == "RMI":
        loss_function = RMILoss(num_classes=num_classes, ignore_index=ignore_index)

    elif name_lf == "wRMI":
        weights = torch.FloatTensor(cfg.DATASET.CLASS_WEIGHTS).cuda()
        loss_function = RMILoss(
            num_classes=num_classes, ignore_index=ignore_index, class_weights=weights
        )

    elif name_lf == "DC":
        loss_function = DiceLoss(mode="multiclass", ignore_index=ignore_index)

    elif name_lf == "DC_CE":
        DC_and_CE = DC_and_CE_loss(
            {"batch_dice": True, "smooth": 0, "do_bg": False},
            {"ignore_index": ignore_index},
            ignore_label=ignore_index,
        )
        loss_function = lambda pred, gt: DC_and_CE(pred.clone(), gt[:, None])
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
    return loss_function
