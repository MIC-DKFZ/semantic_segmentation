import os

import torch
from torchmetrics import Metric
import numpy as np

class ConfusionMatrix(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
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
        torch.save(self.mat.cpu(), path)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

class IoU(ConfusionMatrix):
    #def compute(self):
    #    IoU = self.mat.diag() / (self.mat.sum(1) + self.mat.sum(0) - self.mat.diag())
    #    IoU[IoU.isnan()] = 0
    #    mIoU = IoU.mean()
    #    return IoU, mIoU

    def update(self, gt, pred):
        gt=gt.flatten().detach().cpu()
        pred=pred.argmax(1).flatten().detach().cpu()

        n = self.num_classes

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]

            temp_mat=torch.bincount(inds, minlength=n ** 2).reshape(n, n).to(self.mat)
            IoU,_=self.compute_IoU(temp_mat)
            self.mat += temp_mat
            return IoU

    def compute(self):
        IoU, mIoU = self.compute_IoU(self.mat)
        return IoU#, mIoU

    def compute_IoU(self, mat):
        IoU = mat.diag() / (mat.sum(1) + mat.sum(0) - mat.diag())
        IoU[IoU.isnan()] = 0
        mIoU = IoU.mean()
        return IoU, mIoU

class per_image_Dice(ConfusionMatrix):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(num_classes=num_classes,dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("per_image", default= [], dist_reduce_fx="sum")

    def update(self, gt, pred):
        gt=gt.flatten().detach().cpu()
        pred=pred.argmax(1).flatten().detach().cpu()

        n = self.num_classes

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]

            temp_mat=torch.bincount(inds, minlength=n ** 2).reshape(n, n).to(self.mat)
            dice= self.compute_Dice(temp_mat)
            self.per_image.append(dice.detach().cpu())
            #self.mat += temp_mat

    def compute(self):
        print(self.per_image)
        mDice=np.mean(self.per_image)
        return [],mDice

    def compute_Dice(self,mat):
        Dice = 2*mat.diag() / (mat.sum(1) + mat.sum(0))
        Dice[Dice.isnan()] = 0
        mDice = Dice.mean()
        return mDice

    def save(self, path, name=None):
        return

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()