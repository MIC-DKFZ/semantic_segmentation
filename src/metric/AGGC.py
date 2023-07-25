import numpy as np
import hydra
from omegaconf import DictConfig
import torch
from torchmetrics import Metric, MetricCollection
from src.metric.confmat import ConfusionMatrix


class MetricModule_AGGC(MetricCollection):
    def __init__(self, config: DictConfig, **kwargs) -> None:
        """
        Init MetricModule as subclass of MetricCollection
        Can directly be initialized with the config

        Parameters
        ----------
        config: DictConfig
        kwargs :
            arguments passed to super class (MetricCollection)
        """

        metrics = {}
        for name, m_conf in config.items():
            if m_conf is not None:
                metric = hydra.utils.instantiate(m_conf)
                metrics[name] = metric
        super().__init__(metrics, compute_groups=False, **kwargs)

    def compute(self):
        dic = super().compute()
        # print(dic)
        wF1 = dic["wF1_Subset1"] * 0.6 + dic["wF1_Subset2"] * 0.2 + dic["wF1_Subset3"] * 0.2
        dic["wF1"] = wF1
        return dic


class Dice_AGGC2022(ConfusionMatrix):
    """
    Metric for AGGC2022 Challenge
    Subclass of ConfusionMatrix
    Computing the weighted F1/Dice Score over the 5 Classes of interest (no score for backroung)
    Adapted from: https://github.com/jydada/AGGC2022/blob/main/eval.py
    """

    def compute(self):
        conf_mat = self.mat
        Stroma_Recall = conf_mat[1, 1] / torch.sum(conf_mat[1, :])
        Normal_Recall = conf_mat[2, 2] / torch.sum(conf_mat[2, :])
        G3_Recall = conf_mat[3, 3] / torch.sum(conf_mat[3, :])
        G4_Recall = conf_mat[4, 4] / torch.sum(conf_mat[4, :])
        G5_Recall = conf_mat[5, 5] / torch.sum(conf_mat[5, :])

        Stroma_Pre = conf_mat[1, 1] / (torch.sum(conf_mat[:, 1]) - conf_mat[0, 1])
        Normal_Pre = conf_mat[2, 2] / (torch.sum(conf_mat[:, 2]) - conf_mat[0, 2])
        G3_Pre = conf_mat[3, 3] / (torch.sum(conf_mat[:, 3]) - conf_mat[0, 3])
        G4_Pre = conf_mat[4, 4] / (torch.sum(conf_mat[:, 4]) - conf_mat[0, 4])
        G5_Pre = conf_mat[5, 5] / (torch.sum(conf_mat[:, 5]) - conf_mat[0, 5])

        F1_Stroma = 2 * Stroma_Pre * Stroma_Recall / (Stroma_Pre + Stroma_Recall)
        F1_Normal = 2 * Normal_Pre * Normal_Recall / (Normal_Pre + Normal_Recall)
        F1_G3 = 2 * G3_Pre * G3_Recall / (G3_Pre + G3_Recall)
        F1_G4 = 2 * G4_Pre * G4_Recall / (G4_Pre + G4_Recall)
        F1_G5 = 2 * G5_Pre * G5_Recall / (G5_Pre + G5_Recall)

        if F1_Stroma.isnan():
            F1_Stroma = 0.0
        if F1_Normal.isnan():
            F1_Normal = 0.0
        if F1_G3.isnan():
            F1_G3 = 0.0
        if F1_G4.isnan():
            F1_G4 = 0.0
        if F1_G5.isnan():
            F1_G5 = 0.0

        Weighted_average_F1score = (
            0.25 * F1_G3 + 0.25 * F1_G4 + 0.25 * F1_G5 + 0.125 * F1_Normal + 0.125 * F1_Stroma
        )
        return {
            "wF1": Weighted_average_F1score,
            "F1_Stroma": F1_Stroma,
            "F1_Normal": F1_Normal,
            "F1_G3": F1_G3,
            "F1_G4": F1_G4,
            "F1_G5": F1_G5,
        }


class Dice_AGGC2022_subsets(Dice_AGGC2022):
    def __init__(self, subset, **kwargs):
        self.subset = subset
        super().__init__(**kwargs)

    def update(self, pred: torch.Tensor, gt: torch.Tensor, subsets: str) -> None:
        # print(subset, self.subset, subset == self.subset)
        # print(torch.where(subset == self.subset, True, False))
        if self.subset in subsets:
            if np.all(subsets == self.subset):
                super().update(pred, gt)
            else:
                indices = np.where(subsets == self.subset)[0]
                # indices = indices[1:]
                # indices = torch.where(subset == self.subset)
                # print(indices)
                # print(indices, pred.shape, gt.shape)
                # if indices != []:
                indices = torch.tensor(indices, device=pred.device)
                pred = torch.index_select(pred, 0, indices)
                gt = torch.index_select(gt, 0, indices)

                # print(pred.shape, gt.shape)
                # if pred != []:
                super().update(pred, gt)

    def compute(self):
        # print(self.subset, self.mat)
        out = super(Dice_AGGC2022_subsets, self).compute()
        out = {k + "_" + self.subset: v for k, v in out.items()}
        return out
