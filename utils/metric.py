import os
import numpy as np
import hydra
import torch
from torchmetrics import Metric, MetricCollection, F1Score
from torchmetrics.utilities.data import dim_zero_cat
from utils.utils import get_logger

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay

log = get_logger(__name__)


class MetricModule(MetricCollection):
    ### subclass of MetricCollection which can directly be initialized with the config ###
    def __init__(self, config, **kwargs):
        metrics = {}
        for name, m_conf in config.items():
            if m_conf is not None:
                metric = hydra.utils.instantiate(m_conf)
                metrics[name] = metric

        super().__init__(metrics, **kwargs)

    def save_state_variable(self, logger, epoch):
        for name, met in self.items():
            if hasattr(met, "save_state_variable"):
                met.save_state_variable(logger, epoch)


class ConfusionMatrix(Metric):
    def __init__(self, num_classes: int) -> None:
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_classes = num_classes
        self.add_state(
            "mat",
            default=torch.zeros((num_classes, num_classes), dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.c = 0

    def get_confmat_for_sample(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = gt.flatten()  # .detach()#.cpu()
        pred = pred.argmax(1).flatten()  # .detach()#.cpu()
        n = self.num_classes

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            confmat = torch.bincount(inds, minlength=n**2).reshape(n, n)
        return confmat

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        # with torch.no_grad():
        confmat = self.get_confmat_for_sample(pred, gt)
        self.mat += confmat  # .to(self.mat)

    # def save_state_variable(self, logger, epoch):
    #    df_cm = pd.DataFrame(
    #        self.mat.detach().cpu().numpy() / self.mat.detach().cpu().numpy().sum(),
    #        index=range(self.num_classes),
    #        columns=range(self.num_classes),
    #    )
    #    plt.figure(figsize=(10, 7))
    #    # fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
    #    fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
    #    plt.close(fig_)#

    #    logger.experiment.add_figure(
    #        "ConfusionMatrix/confmat",
    #        fig_,
    #        epoch,
    #    )
    #    logger.experiment.add_image(
    #        tag="ConfusionMatrix/raw_confmat",
    #        img_tensor=self.mat,
    #        dataformats="HW",
    #        global_step=epoch,
    #    )
    # return {"tag": "ConfMat", "img_tensor": self.mat, "dataformats": "HW"}
    # return {"tag": "ConfMat", "text_string": self.mat.item()}
    # return {"tag": "ConfMat", "mat": self.mat}  # , "dataformats": "HW"}

    # def save(self, logger):  # , path, name=None):
    #    logger.experiment.add_image("ConfMat", self.mat, dataformats="HW")  # , global_step=self.c)
    #    self.c += 1
    #    # if name != None:
    #    #    name = "ConfusionMatrix_" + name + ".pt"
    #    # else:
    #    #    name = "ConfusionMatrix.pt"
    #    # path = os.path.join(path, name)
    #    # torch.save(self.mat.detach().cpu(), path)


class IoU(ConfusionMatrix):
    def __init__(self, per_class=False, labels=None, ignore_class=None, **kwargs):
        super().__init__(**kwargs)
        self.per_class = per_class
        self.ignore_class = ignore_class

        if self.per_class:
            if labels is None:
                labels = np.arange(self.num_classes).astype(str)
            if ignore_class is not None:
                labels = np.delete(labels, ignore_class)
            self.labels = ["IoU_" + label for label in labels]

    def get_iou_from_mat(self, confmat: torch.Tensor) -> torch.Tensor:

        # if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
        #    confmat[self.ignore_class] = 0.0

        intersection = confmat.diag()
        IoU = intersection / (confmat.sum(1) + confmat.sum(0) - intersection)

        if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
            IoU = torch.cat((IoU[: self.ignore_class], IoU[self.ignore_class + 1 :]))
        IoU[IoU.isnan()] = 0
        return IoU

    def compute(self):
        IoU = self.get_iou_from_mat(self.mat.clone())
        mIoU = IoU.mean()
        if self.per_class:
            IoU = {self.labels[i]: IoU[i] for i in range(len(IoU))}
            IoU["mIoU"] = mIoU
            return IoU
        else:
            return mIoU


# TestWise
class Dice(F1Score):
    def __init__(
        self,
        num_classes,
        per_class=False,
        labels=None,
        mdmc_average="global",
        multiclass=True,
        ignore_index=None,
        **kwargs
    ):
        # self.per_class=per_class
        # if self.per_class:
        self.per_class = per_class
        self.ignore_class = ignore_index

        if self.per_class:
            kwargs["average"] = None
        super().__init__(
            num_classes=num_classes,
            mdmc_average=mdmc_average,
            multiclass=multiclass,
            ignore_index=ignore_index,
            **kwargs
        )

        if self.per_class:
            if labels is None:
                labels = np.arange(self.num_classes).astype(str)
            if self.ignore_class is not None:
                labels = np.delete(labels, self.ignore_class)
            self.labels = ["Dice_" + label for label in labels]

    def update(self, preds, target):
        target = target.flatten()  # .detach()  # .cpu()
        preds = preds.argmax(1).flatten()  # .detach()  # .cpu()
        k = (target >= 0) & (target < self.num_classes)
        preds = preds[k]
        target = target[k]
        return super().update(preds, target)

    def compute(self):
        Dice = super().compute()
        if self.per_class:
            Dice[Dice.isnan()] = 0.0
            mDice = Dice.mean()
            Dice = {self.labels[i]: Dice[i] for i in range(len(Dice))}
            Dice["mDice"] = mDice
            return Dice
        else:
            return Dice


#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################


class Dice_(ConfusionMatrix):
    def __init__(self, per_class=False, labels=None, ignore_class=None, **kwargs):
        super().__init__(**kwargs)
        self.per_class = per_class
        self.ignore_class = ignore_class

        if self.per_class:
            if labels is None:
                labels = np.arange(self.num_classes).astype(str)
            if ignore_class is not None:
                labels = np.delete(labels, ignore_class)
            self.labels = ["Dice_" + label for label in labels]

    def get_dice_from_mat(self, confmat: torch.Tensor) -> torch.Tensor:
        if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
            confmat[self.ignore_class] = 0.0
        intersection = confmat.diag()
        sum = confmat.sum(1) + confmat.sum(0)
        dice = 2 * intersection / sum

        if self.ignore_class is not None and 0 <= self.ignore_class < self.num_classes:
            dice = torch.cat((dice[: self.ignore_class], dice[self.ignore_class + 1 :]))

        dice[dice.isnan()] = 0
        return dice

    def compute(self):
        # dice = self.get_dice_from_mat(self.mat.clone())
        # mdice = dice.mean()
        dice = self.get_dice_from_mat(self.mat.clone())
        mdice = dice.mean()
        if self.per_class:
            # print(dice)
            dice = {self.labels[i]: dice[i] for i in range(len(dice))}
            dice["mDice"] = mdice
            return dice
        else:
            return mdice
        # if self.per_class:
        #    IoU = {self.labels[i]: IoU[i] for i in range(len(IoU))}
        #    IoU["mIoU"] = mIoU
        #    return IoU
        # else:
        #    return mIoU


class binary_per_image_Dice(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.num_classes = num_classes
        self.add_state("per_image", default=[], dist_reduce_fx="cat")

    def update(self, pred, gt):
        # print(gt.shape, pred.shape)
        # print(gt.shape, pred.argmax(1).shape)
        gt = gt.flatten()  # .detach().cpu()
        pred = pred.argmax(1).flatten()  # .detach().cpu()

        with torch.no_grad():
            gt_s = gt.sum()
            pred_s = pred.sum()
            if gt_s == 0 and pred_s == 0:
                dice = None
            elif gt_s == 0 and pred_s != 0:
                dice = torch.tensor(0, device=pred.device)
                self.per_image.append(dice)
            else:
                # print(gt.shape,pred.shape)
                dice = (2 * (gt * pred).sum() + 1e-15) / (gt_s + pred_s + 1e-15)
                self.per_image.append(dice)

    def compute(self):

        self.per_image = dim_zero_cat(self.per_image)

        # log.info("Not None samples %s", len(self.per_image))
        mDice = self.per_image.clone().detach().mean()

        return (mDice,)


#### Only Testing - no working code ####
class per_image_Dice(Metric):
    # only testing code
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.num_classes = num_classes
        self.add_state("per_image", default=[], dist_reduce_fx="cat")

    def update(self, gt, pred):

        gt = gt.flatten()  # .detach().cpu()
        pred = pred.argmax(1).flatten()  # .detach().cpu()

        with torch.no_grad():

            k = (gt >= 0) & (gt < self.num_classes)
            gt = gt[k]
            pred = pred[k]

            gt = torch.nn.functional.one_hot(gt, self.num_classes)
            pred = torch.nn.functional.one_hot(pred, self.num_classes)

            # dice2 = tmF.dice_score(gt, pred)
            dice = (2 * (gt * pred).sum() + 1e-15) / (gt.sum() + pred.sum() + 1e-15)

            self.per_image.append(dice)

    def compute(self):
        self.per_image = dim_zero_cat(self.per_image)
        # print("CAT_TEST",len(self.per_image),self.control_count)

        # mDice=self.per_image.mean()
        # mDice=torch.tensor(self.per_image).mean()
        mDice = self.per_image.clone().detach().mean()

        # print("control",self.control_dice/self.control_count)
        return mDice  # , None
