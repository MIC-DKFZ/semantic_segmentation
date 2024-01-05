# from src.models.backbones.swin import get_swin_model
import torch

# import logging
# from torch import nn
from mmseg.models import build_segmentor

# from mmseg.models import SegDataPreProcessor
from importlib import import_module

module = import_module(f"mmseg.utils")
module.register_all_modules(True)

# from mmcv.utils import Config
from mmengine.config import Config


class MMSegWrapper(torch.nn.Module):
    def __init__(self, config_path, num_classes, pretrained=True):
        super(MMSegWrapper, self).__init__()
        cfg = Config.fromfile(config_path)
        cfg.model.decode_head.num_classes = num_classes
        self.model = build_segmentor(cfg.model)
        if pretrained:
            self.model.init_weights()

    def forward(self, x):
        batch_img_metas = [
            dict(
                ori_shape=x.shape[2:],
                img_shape=x.shape[2:],
                pad_shape=x.shape[2:],
                padding_size=[0, 0, 0, 0],
            )
        ] * x.shape[0]

        feat = self.model.extract_feat(x)
        seg_logits = self.model.decode_head.predict(feat, batch_img_metas, {"mode": "whole"})
        return {"out": seg_logits}


def get_model(path, num_classes, pretrained=True):

    return MMSegWrapper(path, num_classes, pretrained)


if __name__ == "__main__":
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample

    #
    # path = "src/models/mmseg_configs/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py"
    path = "mmseg_configs/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py"
    path = "mmseg_configs/fcn_hr48_4xb2-160k_cityscapes-512x1024.py"
    path = "mmseg_configs/SegNeXt_l.py"
    path = "mmseg_configs/Mask2Former_b.py"

    # model = get_model(path, 19)
    # img = torch.rand((4, 3, 100, 100))
    #
    # pred = model(img)
    # print(pred["out"].shape)

    num_classes = 19
    img = torch.rand((4, 3, 100, 100))

    img_meta = dict(img_shape=(4, 4, 3), pad_shape=(4, 4, 3))
    data_sample = SegDataSample()

    gt_segmentations = PixelData(metainfo=img_meta)
    gt_segmentations.data = torch.randint(0, num_classes, (100, 100))
    data_sample.gt_sem_seg = gt_segmentations

    cfg = Config.fromfile(path)
    cfg.model.decode_head.num_classes = num_classes
    model = build_segmentor(cfg.model)
    pred = model(img, [data_sample, data_sample, data_sample, data_sample], mode="loss")
    print(torch.mean(torch.tensor(list(pred.values()))))

    #
    # model = MMSegWrapper(path)
    # # print(model)
    # model.eval()

    # x = model.extract_feat(img)
    # print(x[0].shape)
    # x = model.decode_head(x, batch_img_metas, model.test_cfg)  # , batch_data_samples=[])
    # print(len(x[0]))
    # print(x[0][0].shape)
    # print(x[0][1].shape)
    # print(x[0][2].shape)
    # print(x[0][3].shape)
    # pred = model(img)  # , mode="predict")
    # print(len(pred))
    # print(pred.seg_logits.shape)
    # print(pred[1].shape)
    # print(pred[2].shape)
    # print(model)
    # print("P", pred)

    # model = get_swin_model()
    # print(model)
    # img = torch.rand((3, 3, 100, 100))
    # out = model(img)
    # print(out.keys())
    # print(out["res2"].shape)
    # print(out["res3"].shape)
    # print(out["res4"].shape)
    # print(out["res5"].shape)
