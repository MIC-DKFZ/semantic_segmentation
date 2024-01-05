"""
Code slightly adapted and mainly from:
https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segnext
"""
checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth"  # noqa
ham_norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="MSCAN",
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="BN", requires_grad=True),
    ),
    decode_head=dict(
        type="LightHamHead",
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=None,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
