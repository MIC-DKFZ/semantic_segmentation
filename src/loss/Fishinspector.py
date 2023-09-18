import torch
import torch.nn
import torch.nn as nn
import numpy as np


class BCEWithLogitsLossPartlyLabeled(torch.nn.BCEWithLogitsLoss):
    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, predicted, target, map=None):
        loss = super().forward(predicted, target.to(torch.float16))
        loss = torch.mean(loss, dim=(-2, -1))
        if map is not None:
            return torch.sum(loss * map)
        else:
            return torch.sum(loss)


class BCE_PL(nn.Module):
    def __init__(self):
        super(BCE_PL, self).__init__()
        self.lf = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predicted, target, map=None):
        loss = self.lf(predicted, target)
        loss = torch.mean(loss, dim=(-2, -1))
        if map is not None:
            return torch.sum(loss * map)
        else:
            return torch.sum(loss)


class BCE_PLv2(nn.Module):
    def __init__(self):
        super(BCE_PLv2, self).__init__()
        self.lf = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predicted, target, map=None):

        loss = self.lf(predicted, target)
        loss = torch.mean(loss, dim=(-2, -1))

        if map is not None:
            return torch.sum(loss * map) / len(predicted)
        else:

            return torch.sum(loss) / len(predicted)


class SoftDiceLoss_Multilabel_PL(nn.Module):
    def __init__(
        self,
        apply_nonlin=None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
        clip_tp: float = None,
    ):
        """ """
        super(SoftDiceLoss_Multilabel_PL, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, mask, loss_mask=None):

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = [2, 3]
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp, max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        print(torch.any(torch.isnan(dc)))
        dc = torch.nanmean(dc[mask])  # .mean()
        # dc = dc[mask].sum() / (mask.sum() + 1e-8)  # .mean()

        return -dc


def sum_tensor(inp: torch.Tensor, axes, keepdim: bool = False) -> torch.Tensor:
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for i in range(2, len(tp.shape))]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp**2
        fp = fp**2
        fn = fn**2
        tn = tn**2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn
