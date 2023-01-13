import os

import torch.nn as nn
import torch.nn.functional as F
from models.backbones.mscan import MSCAN
import torch
from src.utils import get_logger

log = get_logger(__name__)


class ConvModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, conv_cfg=None, norm_cfg=None, act_cfg=None
    ):
        super(ConvModule, self).__init__()
        # print(in_channels, out_channels, kernel_size, conv_cfg, norm_cfg, act_cfg)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.norm = norm_cfg is not None
        self.act = act_cfg is not None

        if self.norm:
            self.gn = nn.GroupNorm(norm_cfg["num_groups"], num_channels=out_channels)
            # self.gn = nn.BatchNorm2d(norm_cfg["num_groups"], num_channels=out_channels)

        if self.act:
            self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.gn(x)

        if self.act:
            x = self.activate(x)
        return x


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault("SPATIAL", True)

        self.S = args.setdefault("MD_S", 1)
        self.D = args.setdefault("MD_D", 512)
        self.R = args.setdefault("MD_R", 64)

        self.train_steps = args.setdefault("TRAIN_STEPS", 6)
        self.eval_steps = args.setdefault("EVAL_STEPS", 7)

        self.inv_t = args.setdefault("INV_T", 100)
        self.eta = args.setdefault("ETA", 0.9)

        self.rand_init = args.setdefault("RAND_INIT", True)

        # print("spatial", self.spatial)
        # print("S", self.S)
        # print("D", self.D)
        # print("R", self.R)
        # print("train_steps", self.train_steps)
        # print("eval_steps", self.eval_steps)
        # print("inv_t", self.inv_t)
        # print("eta", self.eta)
        # print("rand_init", self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer("bases", bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), norm_cfg=None, **kwargs):
        super().__init__()

        self.ham_in = ConvModule(ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


class LightHamHead(nn.Module):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO:
        Add other MD models (Ham).
    """

    def __init__(
        self,
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=512,
        ham_channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        act_cfg=dict(type="ReLU"),
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        ham_kwargs=dict(),
        align_corners=False,
        **kwargs
    ):
        super(LightHamHead, self).__init__()
        self.ham_channels = ham_channels
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.channels = channels
        self.in_index = in_index
        self.align_corners = align_corners

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.conv_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.hamburger = Hamburger(ham_channels, ham_kwargs, norm_cfg, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, inputs):
        """Forward function."""

        inputs = [inputs[i] for i in self.in_index]

        inputs = [
            F.interpolate(
                level, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)

        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        return output


##########################################
class MMSeg_Model(nn.Module):
    def __init__(self, cfg):
        super(MMSeg_Model, self).__init__()
        # self.MSCAN(**cfg.MSCAN)
        self.backbone = MSCAN(**cfg.MSCAN)
        # print(self.backbone)
        # print(cfg.LightHamHead)
        self.decode_head = LightHamHead(**cfg.LightHamHead)

    def forward(self, x):
        x_size = x.size(2), x.size(3)
        x = self.backbone(x)

        x = self.decode_head(x)
        # if torch.isnan(x).any():
        #    print("NAN prediction Head")
        x = F.interpolate(
            x, size=x_size, mode="bilinear", align_corners=self.decode_head.align_corners
        )
        # if torch.isnan(x).any():
        #    print("NAN prediction")
        # quit()
        x = {"out": x}
        return x

    def load_weights(self, pretrained):

        if os.path.isfile(pretrained):

            pretrained_dict = torch.load(pretrained, map_location={"cuda:0": "cpu"})
            log.info("Loading pretrained weights {}".format(pretrained))
            # print(
            #    "Weights", torch.min(pretrained_dict.values()), torch.max(pretrained_dict.values())
            # )
            # some preprocessing
            if "state_dict" in pretrained_dict.keys():
                pretrained_dict = pretrained_dict["state_dict"]
            model_dict = self.state_dict()
            update_dict = {}

            for p, pv in pretrained_dict.items():
                # print(p, torch.min(pv), torch.max(pv))
                for m, mv in model_dict.items():
                    if p in m:  # and torch.max(pv) < 302806:
                        # if pv.dtype != mv.dtype:
                        # print(p, m)
                        update_dict[m] = pv  # / 255
                        continue

            no_match = set(model_dict) - set(update_dict)
            # check if shape of pretrained weights match to the model
            update_dict = {k: v for k, v in update_dict.items() if v.shape == model_dict[k].shape}
            shape_mismatch = (set(model_dict) - set(update_dict)) - no_match

            # log info about weights which are not found and weights which have a shape mismatch
            if len(no_match):
                num = len(no_match)
                if num >= 5:
                    no_match = list(no_match)[:5]
                    no_match.append("...")
                log.warning(
                    "No pretrained Weights found for {} of {} layers: {}".format(
                        num, len(model_dict.keys()), no_match
                    )
                )
            if len(shape_mismatch):
                num = len(shape_mismatch)
                if num >= 5:
                    shape_mismatch = list(shape_mismatch)[:5]
                    shape_mismatch.append("...")
                log.warning("Shape Mismatch for {} layers: {}".format(num, shape_mismatch))

            # load weights
            model_dict.update(update_dict)
            self.load_state_dict(model_dict)
            del model_dict, pretrained_dict, update_dict
            log.info("Weights successfully loaded")
        else:
            # raise NotImplementedError("No Pretrained Weights found for {}".format_map(pretrained))
            # print(os.getcwd())
            raise NotImplementedError("No Pretrained Weights found for {}".format(pretrained))


def get_seg_model(cfg):
    model = MMSeg_Model(cfg)

    if cfg.pretrained:
        model.load_weights(cfg.pretrained_weights)
    return model


if __name__ == "__main__":
    from omegaconf import OmegaConf

    path = "/home/l727r/Desktop/Semantic_Segmentation/config/model/SegNeXt_test.yaml"
    cfg = OmegaConf.load(path)
    print(cfg.model.cfg.MSCAN)
    print(cfg.model.cfg.LightHamHead)
    model = get_seg_model(cfg.model.cfg).cuda()
    input = (torch.rand((2, 3, 512, 1024))).cuda()
    model.eval()
    with torch.no_grad():
        pred = model(input)["out"]
        print(pred.shape)
        if torch.isnan(pred).any():
            print("OH")
