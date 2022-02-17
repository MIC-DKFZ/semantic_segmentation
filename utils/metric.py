import os

import torch
from torchmetrics import Metric
import torchmetrics.functional as tmF
import numpy as np
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
    #def __init__(self, num_classes,dist_sync_on_step=False):
        #super().__init__( num_classes,dist_sync_on_step)
        #self.labels=None#labels
        #print(labels)

    def compute(self):

        mIoU,IoU_class= self.compute_IoU(self.mat)
        #print(IoU_class)

        #dic_score = {}
        #if self.labels!=None:
        #    for i,l in zip(IoU_class,self.labels):
        #        dic_score[l]= "%.4f" % i.item()
        #else:
        #    for i,l in zip(IoU_class,range(0,len(IoU_class))):
        #        dic_score[str(l)]="%.4f" % i.item()
        #log.info(dic_score)

        #for id, sc in enumerate(IoU_class):
        #    if hasNotEmptyAttr(self.config.DATASET, "CLASS_LABELS"):
        #        dic_score[str(id) + "-" + self.config.DATASET.CLASS_LABELS[id]] = "%.4f" % sc.item()
        #    else:
        #        dic_score[id] = "%.4f" % sc.item()
        # log.info(dic_score)
        #print(mIoU)
        #print(mIoU.item())
        #dic={"IoU per Class":dic_score}
        return mIoU#,dic#, dic_score#dic_score#{"P":mIoU,"R":mIoU} #IoU_class

    def compute_IoU(self, mat):

        IoU = mat.diag() / (mat.sum(1) + mat.sum(0) - mat.diag())
        IoU[IoU.isnan()] = 0

        mIoU = IoU.mean()
        return mIoU, IoU




class per_image_Dice(Metric):
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
        return mDice, None

    def save(self, path, name=None):
        return

class per_image_Dice_class(Metric):
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
            #print(pred.shape,gt.shape)
            gt = torch.nn.functional.one_hot(gt, self.num_classes)
            pred = torch.nn.functional.one_hot(pred, self.num_classes)
            #print(pred.shape, gt.shape)
            gt = gt[:, 1:]
            pred = pred[:,1:]
            #print(pred.shape, gt.shape)
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
        return mDice, None

    def save(self, path, name=None):
        return
    #def reset(self):
    #    print("Rest")
    #    self.per_image=torch.tensor([])