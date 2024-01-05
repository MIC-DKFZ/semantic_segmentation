_base_ = ["./Mask2Former_base.py"]
pretrained = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth"  # noqa
depths = [2, 2, 6, 2]
model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768]),
)
