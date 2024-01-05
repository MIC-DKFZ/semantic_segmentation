_base_ = ["./Mask2Former_t.py"]
pretrained = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth"  # noqa

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=12,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
)
