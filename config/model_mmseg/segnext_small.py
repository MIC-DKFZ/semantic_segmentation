"""
Source:
https://github.com/Visual-Attention-Network/SegNeXt/blob/main/local_configs/segnext/small
"""

_base_ = [
    "mscan.py",
]
# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
ham_norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        init_cfg=dict(type="Pretrained", checkpoint="pretrained/mscan_s.pth"),
    ),
    decode_head=dict(
        type="LightHamHead",
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(MD_R=16),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    # needed for mmseg_old, not used here
    train_cfg=dict(),
    test_cfg=dict(),
)
