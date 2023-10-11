import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple

# from mmseg_old.models.builder import BACKBONES

# from mmcv.cnn import build_norm_layer
# from mmcv.runner import BaseModule
# from mmcv.cnn.bricks import DropPath
# from mmcv.cnn.utils.weight_init import constant_init, normal_init, trunc_normal_init
def build_norm_layer(cfg, num_features):
    return nn.BatchNorm2d(num_features)
    # return nn.SyncBatchNorm(num_features)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type="SyncBN", requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),
            build_norm_layer(norm_cfg, out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),
            build_norm_layer(norm_cfg, out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        # print("IN", x.dtype)
        shorcut = x.clone()
        # if torch.isnan(shorcut).any():
        #     print("NAN SPA short")
        x = self.proj_1(x)
        # if torch.isnan(x).any():
        #     print("NAN SPA proj1")
        x = self.activation(x)
        # if torch.isnan(x).any():
        #     print("NAN SPA act")
        # print("B", torch.min(x), torch.max(x))
        x = self.spatial_gating_unit(x)
        # if torch.isnan(x).any():
        #     print("NAN SPA sgu")
        # print(x)
        # print("I", torch.min(x), torch.max(x))
        # x_org = x.clone()
        x = self.proj_2(x)
        # print("O", torch.min(x), torch.max(x))
        # print("Out", x.dtype)
        # if torch.isnan(x).any():
        #     print("NAN SPA proj2")
        # print(x_org)
        # print(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
    ):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        if torch.isnan(x).any():
            print("NAN Block Inp")
        # x = x + self.drop_path(
        #     self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        # )
        l1 = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
        # if torch.isnan(l1).any():
        #     print("NAN Block l1")
        n = self.norm1(x)
        # if torch.isnan(n).any():
        #     print("NAN Block n")
        at = self.attn(n)
        # if torch.isnan(at).any():
        #     print("NAN Block At")
        dr = self.drop_path(l1 * at)
        # if torch.isnan(dr).any():
        #     print("NAN Block dr")
        x = x + dr
        # if torch.isnan(x).any():
        #     print("NAN Block Drop1")
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        # if torch.isnan(x).any():
        #     print("NAN Block Drop2")
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=7,
        stride=4,
        in_chans=3,
        embed_dim=768,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = build_norm_layer(norm_cfg, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class MSCAN(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[4, 4, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.0,
        depths=[3, 4, 6, 3],
        num_stages=4,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        pretrained=None,
        init_cfg=None,
    ):
        super(MSCAN, self).__init__()
        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be set at the same time"
        if isinstance(pretrained, str):
            warnings.warn(
                'DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError("pretrained must be a str or None")

        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg,
                )

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                        norm_cfg=norm_cfg,
                    )
                    for j in range(depths[i])
                ]
            )
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            # print("Korrekt", patch_embed)
            # for x_i in x:
            #     if torch.isnan(x_i).any():
            #         print("NAN in enb", i)
            # print(patch_embed)

            for blk in block:
                x = blk(x, H, W)
            # for x_i in x:
            #     if torch.isnan(x_i).any():
            # print("NAN in block", i)
            # print(block)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
