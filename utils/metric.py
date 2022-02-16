import os

import torch
from torchmetrics import Metric
import torchmetrics.functional as tmF
import numpy as np
from torchmetrics.utilities.data import dim_zero_cat

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

    def compute(self):
        IoU_class, mIoU = self.compute_IoU(self.mat)
        return mIoU, IoU_class

    def compute_IoU(self, mat):
        IoU = mat.diag() / (mat.sum(1) + mat.sum(0) - mat.diag())
        IoU[IoU.isnan()] = 0
        mIoU = IoU.mean()
        return IoU, mIoU




class per_image_Dice(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.num_classes = num_classes
        #self.add_state("mat", default=torch.zeros((num_classes, num_classes), dtype=torch.int64), dist_reduce_fx="sum")
        #self.add_state("per_image", default= torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("per_image", default=[], dist_reduce_fx="cat")
        #self.add_state("control_dice", default=torch.tensor(0.0), dist_reduce_fx="sum")
        #self.add_state("control_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, gt, pred):

        gt = gt.flatten()#.detach().cpu()
        pred = pred.argmax(1).flatten()#.detach().cpu()

        with torch.no_grad():

            #inds = n * gt[k].to(torch.int64) + pred[k]

            #temp_mat=torch.bincount(inds, minlength=n ** 2).reshape(n, n).to(self.mat)
            #dice2= self.compute_Dice(temp_mat)
            #dice = self.mat.diag().sum() / self.mat.sum()
            ##################
            #gt2 = torch.nn.functional.one_hot(gt, self.num_classes)
            #pred2 = torch.nn.functional.one_hot(pred, self.num_classes)

            #dice2 = (2 * (gt2 * pred2).sum() + 1e-15) / (gt2.sum() + pred2.sum() + 1e-15)

            k = (gt >= 0) & (gt < self.num_classes)
            gt = gt[k]
            pred = pred[k]

            gt = torch.nn.functional.one_hot(gt, self.num_classes)
            pred = torch.nn.functional.one_hot(pred, self.num_classes)

            #dice2 = tmF.dice_score(gt, pred)
            dice = (2 * (gt * pred).sum() + 1e-15) / (gt.sum() + pred.sum() + 1e-15)

            #print(self.per_image,dice)
            #print(self.mat)
            #self.per_image = torch.cat((self.per_image,dice.unsqueeze(0).to(self.per_image)))
            self.per_image.append(dice)
            #self.control_count+=1
            #self.control_dice+=dice#.to(self.control_dice)

            #print(len(self.per_image))
            #print(self.per_image)

    def compute(self):
        self.per_image=dim_zero_cat(self.per_image)
        #print("CAT_TEST",len(self.per_image),self.control_count)

        #mDice=self.per_image.mean()
        mDice=torch.tensor(self.per_image).mean()

        #print("control",self.control_dice/self.control_count)
        return mDice, None

    def save(self, path, name=None):
        return

    #def reset(self):
    #    print("Rest")
    #    self.per_image=torch.tensor([])