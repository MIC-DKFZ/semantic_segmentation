_base_ = ["./Mask2Former_t.py"]
pretrained = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth"  # noqa

depths = [2, 2, 18, 2]
model = dict(backbone=dict(depths=depths, init_cfg=dict(type="Pretrained", checkpoint=pretrained)))
