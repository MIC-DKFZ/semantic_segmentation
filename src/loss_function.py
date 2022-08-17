import torch
import torch.nn
from omegaconf import DictConfig
from src.loss.rmi import RMILoss
from src.loss.Dice_Loss import DiceLoss
from src.loss.DC_CE_Loss import DC_and_CE_loss, TopKLoss, DC_and_topk_loss
from src.utils import has_not_empty_attr


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
    Lossfunction
    """
    num_classes = cfg.DATASET.NUM_CLASSES
    ignore_index = (
        cfg.DATASET.IGNORE_INDEX if has_not_empty_attr(cfg.DATASET, "IGNORE_INDEX") else -100
    )
    if name_lf == "CE":
        loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )  # , label_smoothing=0.1)
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


"""
# Some testing with weighting Batch samples
class CrossEntropyLoss_AGGC(torch.nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        self.subset_weights = {"Subset1": 1.0, "Subset2": 0.33, "Subset3": 0.33}
        # self.subset_weights = {"Subset1": 1, "Subset2": 1, "Subset3": 1}
        super().__init__(**kwargs)

    def forward(self, input: Tensor, target: Tensor, subset: [str] = None) -> Tensor:
        loss = super(CrossEntropyLoss_AGGC, self).forward(input, target)
        # loss = torch.mean(loss, dim=1)
        # print(sum(self.weight))
        # loss.sum() / self.weight[target].sum()
        if self.weight is not None:
            weights = (
                torch.tensor([self.subset_weights[s] for s in subset], device=loss.device)
                .unsqueeze(1)
                .unsqueeze(2)
            )
            loss *= weights
            loss = loss.sum() / self.weight[target].sum()

        else:
            weights = (
                torch.tensor([self.subset_weights[s] for s in subset], device=loss.device)
                .unsqueeze(1)
                .unsqueeze(2)
            )
            loss *= weights
            loss = loss[target != self.ignore_index].mean()

        return loss
"""
