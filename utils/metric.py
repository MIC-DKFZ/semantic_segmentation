import os
import numpy as np

import torch
from torchmetrics import Metric
import torchmetrics.functional as tmF
from torchmetrics.utilities.data import dim_zero_cat

from utils.utils import hasNotEmptyAttr
from utils.utils import get_logger

log = get_logger(__name__)

class ConfusionMatrix(Metric):
    def __init__(self, num_classes,dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.num_classes = num_classes
        self.add_state("mat", default=torch.zeros((num_classes, num_classes), dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, gt, pred):
        gt=gt.flatten().detach().cpu()
        pred=pred.argmax(1).flatten().detach().cpu()

        n = self.num_classes

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n).to(self.mat)

    def save(self, path, name=None):
        if name != None:
            name="ConfusionMatrix_"+name+".pt"
        else:
            name="ConfusionMatrix.pt"
        path = os.path.join(path, name)
        torch.save(self.mat.detach().cpu(), path)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

class IoU(ConfusionMatrix):

    def compute(self):

        mIoU,_= self.compute_IoU(self.mat)

        return mIoU

    def compute_IoU(self, mat):

        IoU = mat.diag() / (mat.sum(1) + mat.sum(0) - mat.diag())
        IoU[IoU.isnan()] = 0

        mIoU = IoU.mean()
        return mIoU, IoU


class IoU_Class(ConfusionMatrix):
    def __init__(self, num_classes,labels=None,dist_sync_on_step=False):
        super().__init__( num_classes,dist_sync_on_step)
        self.labels=labels

    def compute(self):

        mIoU,IoU_class= self.compute_IoU(self.mat)

        dic_score = {}
        if self.labels!=None:
            for i,l in zip(IoU_class,self.labels):
                dic_score[l]= i
        else:
            for i,l in zip(IoU_class,range(0,len(IoU_class))):
                dic_score[str(l)]=i

        return mIoU,dic_score

    def compute_IoU(self, mat):

        IoU = mat.diag() / (mat.sum(1) + mat.sum(0) - mat.diag())
        IoU[IoU.isnan()] = 0

        mIoU = IoU.mean()
        return mIoU, IoU

class binary_per_image_Dice(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.num_classes = num_classes
        self.add_state("per_image", default=[], dist_reduce_fx="cat")

    def update(self, gt, pred):
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
                dice = (2 * (gt * pred).sum() + 1e-15) / (gt_s + pred_s + 1e-15)
                self.per_image.append(dice)

    def compute(self):

        self.per_image = dim_zero_cat(self.per_image)

        log.info("Not None samples %s", len(self.per_image))
        mDice = self.per_image.clone().detach().mean()

        return mDice, None

#### Only Testing - no working code ####
class per_image_Dice(Metric):
    #only testing code
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.num_classes = num_classes
        self.add_state("per_image", default=[], dist_reduce_fx="cat")

    def update(self, gt, pred):

        gt = gt.flatten()#.detach().cpu()
        pred = pred.argmax(1).flatten()#.detach().cpu()

        with torch.no_grad():

            k = (gt >= 0) & (gt < self.num_classes)
            gt = gt[k]
            pred = pred[k]

            gt = torch.nn.functional.one_hot(gt, self.num_classes)
            pred = torch.nn.functional.one_hot(pred, self.num_classes)

            #dice2 = tmF.dice_score(gt, pred)
            dice = (2 * (gt * pred).sum() + 1e-15) / (gt.sum() + pred.sum() + 1e-15)

            self.per_image.append(dice)

    def compute(self):
        self.per_image=dim_zero_cat(self.per_image)
        #print("CAT_TEST",len(self.per_image),self.control_count)

        #mDice=self.per_image.mean()
        #mDice=torch.tensor(self.per_image).mean()
        mDice=self.per_image.clone().detach().mean()

        #print("control",self.control_dice/self.control_count)
        return mDice#, None
