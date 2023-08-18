"""
------------------------------------------------------------------------------
Code slightly adapted and mainly from:
https://github.com/NVIDIA/semantic-segmentation/tree/main/network
 - https://github.com/NVIDIA/semantic-segmentation/blob/main/network/ocrnet.py
 - https://github.com/NVIDIA/semantic-segmentation/blob/main/network/ocr_utils.py
 - https://github.com/NVIDIA/semantic-segmentation/blob/main/network/utils.py
 - https://github.com/NVIDIA/semantic-segmentation/blob/main/network/mynn.py
        ------------------------------------------------------------------------------
        Copyright 2020 Nvidia Corporation

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this
           list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright notice,
           this list of conditions and the following disclaimer in the documentation
           and/or other materials provided with the distribution.

        3. Neither the name of the copyright holder nor the names of its contributors
           may be used to endorse or promote products derived from this software
           without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
        LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
        CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
        ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
        POSSIBILITY OF SUCH DAMAGE.
        ------------------------------------------------------------------------------
------------------------------------------------------------------------------
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.backbones.hrnet_backbone import get_backbone_model

from src.utils.utils import get_logger
from src.utils.model_utils import update_state_dict

log = get_logger(__name__)

ALIGN_CORNERS = None
INIT_DECODER = False
Norm2d = nn.BatchNorm2d


def BNReLU(ch):
    return nn.Sequential(Norm2d(ch), nn.ReLU())


def scale_as(x, y):
    """
    scale x to the same size as y
    """
    y_size = y.size(2), y.size(3)
    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode="bilinear", align_corners=ALIGN_CORNERS
    )
    return x_scaled


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=ALIGN_CORNERS)


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


#######ASPP############


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise "output stride of {} not supported".format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True),
            )
        )
        # other rates
        for r in rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_dim,
                        reduction_dim,
                        kernel_size=3,
                        dilation=r,
                        padding=r,
                        bias=False,
                    ),
                    Norm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False):
    """
    Create aspp block
    """
    aspp = AtrousSpatialPyramidPoolingModule(
        high_level_ch, bottleneck_ch, output_stride=output_stride
    )
    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch


#######OCR#############


class SpatialGather_Module(nn.Module):
    """
    Aggregate the context features according to the initial
    predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.
    Output:
      The correlation of every class map with every feature map
      shape = [n, num_feats, num_classes, 1]
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(2), probs.size(3)

        # each class image now a vector
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)

        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature
                            maps (save memory cost)
    Return:
        N X C X H X W
    """

    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.in_channels),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(
                input=context, size=(h, w), mode="bilinear", align_corners=ALIGN_CORNERS
            )

        return context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation
    for each pixel.
    """

    def __init__(
        self,
        in_channels,
        key_channels,
        out_channels,
        bottleneck_ch,
        scale=1,
        dropout=0.1,
    ):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels, key_channels, scale)

        self.aspp, aspp_out_ch = get_aspp(in_channels, bottleneck_ch=bottleneck_ch, output_stride=8)
        _in_channels = 2 * in_channels + aspp_out_ch

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            BNReLU(out_channels),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        aspp = self.aspp(feats)
        output = self.conv_bn_dropout(torch.cat([context, aspp, feats], 1))

        return output


class OCR_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """

    def __init__(self, cfg, high_level_ch):
        super(OCR_block, self).__init__()

        ocr_mid_channels = cfg.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = cfg.MODEL.OCR.KEY_CHANNELS
        num_classes = cfg.DATASET.NUM_CLASSES

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            bottleneck_ch=cfg.MODEL.ASPP_BOT_CH,
            scale=1,
            dropout=0.05,
        )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch, kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(
                high_level_ch,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        if INIT_DECODER:
            initialize_weights(
                self.conv3x3_ocr,
                self.ocr_gather_head,
                self.ocr_distri_head,
                self.cls_head,
                self.aux_head,
            )

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


class OCRNetASPP(nn.Module):
    """
    OCR net
    """

    def __init__(self, cfg):
        global ALIGN_CORNERS
        super(OCRNetASPP, self).__init__()
        ALIGN_CORNERS = cfg.MODEL.ALIGN_CORNERS
        # self.criterion = criterion
        # self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.backbone = get_backbone_model(cfg)
        high_level_ch = self.backbone.high_level_ch
        self.aspp, aspp_out_ch = get_aspp(high_level_ch, bottleneck_ch=256, output_stride=8)
        self.ocr = OCR_block(cfg, aspp_out_ch)

    def forward(self, x):
        x_size = x.size(2), x.size(3)
        _, _, high_level_features = self.backbone(x)
        aspp = self.aspp(high_level_features)
        cls_out, aux_out, _ = self.ocr(aspp)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        aux_out = torch.nn.functional.interpolate(
            aux_out, size=x_size, mode="bilinear", align_corners=ALIGN_CORNERS
        )
        cls_out = torch.nn.functional.interpolate(
            cls_out, size=x_size, mode="bilinear", align_corners=ALIGN_CORNERS
        )

        return {"out": cls_out, "aux": aux_out}

    def load_weights(self, pretrained):
        self.load_state_dict(
            update_state_dict(
                pretrained,
                self.state_dict(),
                ["model.", "module.", "backbone."],
                "pretrained OCR ASPP",
            )
        )


def get_seg_model(cfg):
    model = OCRNetASPP(cfg)
    if cfg.MODEL.PRETRAINED:
        model.load_weights(cfg.MODEL.PRETRAINED_WEIGHTS)
    return model
