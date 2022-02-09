import os

import torch
from torchmetrics import Metric

class ConfusionMatrix(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.num_classes = num_classes
        self.add_state("mat", default=torch.zeros((num_classes, num_classes), dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, gt, pred):
        gt=gt.flatten().detach().cpu()
        pred=pred.argmax(1).flatten().detach().cpu()

        n = self.num_classes
        #gt=gt.detach().cpu()
        #pred=pred.detach().cpu()

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n).to(self.mat)

    def compute(self):
        IoU = self.mat.diag() / (self.mat.sum(1) + self.mat.sum(0) - self.mat.diag())
        IoU[IoU.isnan()] = 0
        mIoU = IoU.mean()
        return IoU, mIoU

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