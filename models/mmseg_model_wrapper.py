import os

# import torch
from mmseg.models import build_segmentor
from mmcv.utils import Config

# import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm

import torch.nn as nn
import torch.nn.functional as F
import logging
from src.utils import get_logger, has_not_empty_attr


class MMSeg_Model(nn.Module):
    def __init__(self, file, num_classes=19, root_dir=""):
        super(MMSeg_Model, self).__init__()
        cfg = Config.fromfile(os.path.join(root_dir, file))
        if has_not_empty_attr(cfg.model.backbone, "init_cfg"):
            if has_not_empty_attr(cfg.model.backbone.init_cfg, "checkpoint"):
                cfg.model.backbone.init_cfg.checkpoint = os.path.join(
                    root_dir, cfg.model.backbone.init_cfg.checkpoint
                )
        cfg.model.decode_head.num_classes = num_classes
        self.mmseg_model = build_segmentor(cfg.model)
        # logger = mmcv.utils.get_logger("mmdet")
        # logger = mmcv.utils.get_logger("mmcv")
        # logger.setLevel("WARNING")
        self.mmseg_model.init_weights()
        self.mmseg_model = revert_sync_batchnorm(self.mmseg_model)
        self.mmseg_model.CLASSES = num_classes
        self.align_corners = self.mmseg_model.align_corners
        # quit()
        # mmseg and mmcv messing up with the loggers, so the loggers have to be "repaired" here
        for name in logging.root.manager.loggerDict:
            log = get_logger(name)
            for handler in log.root.handlers:
                if type(handler) is logging.StreamHandler:
                    handler.setLevel(logging.INFO)
        # print(self.mmseg_model)
        # quit()

    def forward(self, x):
        x_size = x.size(2), x.size(3)
        feat = self.mmseg_model.extract_feat(x)
        y = self.mmseg_model.decode_head(feat)
        y = F.interpolate(y, size=x_size, mode="bilinear", align_corners=self.align_corners)

        # if torch.isnan(y).any():
        #     print("NAN prediction")
        y = {"out": y}
        if self.mmseg_model.with_auxiliary_head:
            y_aux = self.mmseg_model.auxiliary_head(feat)
            y = F.interpolate(y_aux, size=x_size, mode="bilinear", align_corners=self.align_corners)
            y["aux"] = y_aux
        return y
