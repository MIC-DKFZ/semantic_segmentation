import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities.data import dim_zero_cat


class binary_Dice(Metric):
    def __init__(self):
        """
        Binary Dice for the EndoCV2022 challenge with ignoring background (class 0)

        update:
        if gt = 0 and pred = 0, irgnore sample
        if gt = 0 amd pred !=0, return 0
        else compute dice score

        compute:
        mean over the results of each sample
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_classes = 2
        self.add_state("per_image", default=[], dist_reduce_fx="cat")

    def update(self, pred, gt):
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
        mDice = self.per_image.mean()

        return mDice
