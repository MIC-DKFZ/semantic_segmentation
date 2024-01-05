_base_ = ["./Mask2Former_t.py"]
pretrained = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"  # noqa

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=depths,
        num_heads=[6, 12, 24, 48],
        window_size=12,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    decode_head=dict(in_channels=[192, 384, 768, 1536]),
)
