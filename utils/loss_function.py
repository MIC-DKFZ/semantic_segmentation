import torch
from utils.loss.rmi import RMILoss
from utils.loss.Dice_Loss import DiceLoss
from utils.loss.DC_CE_Loss import DC_and_CE_loss, TopKLoss, DC_and_topk_loss


def get_loss_function_from_cfg(LOSSFUNCTION, cfg):
    num_classes = cfg.DATASET.NUM_CLASSES
    ignore_index = cfg.DATASET.IGNORE_INDEX
    if LOSSFUNCTION == "CE":
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    elif LOSSFUNCTION == "wCE":
        weights = torch.FloatTensor(cfg.DATASET.CLASS_WEIGHTS).cuda()
        print(weights)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)

    elif LOSSFUNCTION == "RMI":
        loss_function = RMILoss(num_classes=num_classes, ignore_index=ignore_index)

    elif LOSSFUNCTION == "wRMI":
        weights = torch.FloatTensor(cfg.DATASET.CLASS_WEIGHTS).cuda()
        loss_function = RMILoss(
            num_classes=num_classes, ignore_index=ignore_index, class_weights=weights
        )

    elif LOSSFUNCTION == "DC":
        loss_function = DiceLoss(mode="multiclass", ignore_index=ignore_index)

    elif LOSSFUNCTION == "DC_CE":
        DC_and_CE = DC_and_CE_loss(
            {"batch_dice": True, "smooth": 0, "do_bg": False},
            {"ignore_index": ignore_index},
            ignore_label=ignore_index,
        )
        loss_function = lambda pred, gt: DC_and_CE(pred.clone(), gt[:, None])
    elif LOSSFUNCTION == "TOPK":
        TopK = TopKLoss(ignore_index=ignore_index)
        loss_function = lambda pred, gt: TopK(pred.clone(), gt[:, None])
    elif LOSSFUNCTION == "DC_TOPK":
        DC_TopK = DC_and_topk_loss(
            {"batch_dice": True, "smooth": 0, "do_bg": False},
            {"ignore_index": ignore_index},
            ignore_label=ignore_index,
        )
        loss_function = lambda pred, gt: DC_TopK(pred.clone(), gt[:, None])

    return loss_function
