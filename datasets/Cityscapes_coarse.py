import os
import glob
from collections import namedtuple
import logging
log = logging.getLogger(__name__)

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.Cityscapes import Cityscapes_dataset,show_cityscape

class Cityscapes_coarse_dataset(Cityscapes_dataset):
    def __init__(self,root,split="train",transforms=None):

        if isinstance(root, str):
            root_imgs=root
            root_labels=root
        else:
            root_imgs = root.IMAGES
            root_labels = root.LABELS

        if split=="train":
            imgs_path=os.path.join( root_imgs ,"leftImg8bit_trainextra", "leftImg8bit" , "train_extra" , "*" , "*_leftImg8bit.png" )
            masks_path = os.path.join(root_labels, "gtCoarse", "gtCoarse", "train_extra", "*","*_gt*_labelIds_19classes.png")
        elif split=="test" or split=="val":
            imgs_path = os.path.join(root_imgs, "leftImg8bit_trainvaltest", "leftImg8bit", split, "*", "*_leftImg8bit.png")
            masks_path = os.path.join(root_labels, "gtFine_trainvaltest", "gtFine", split, "*", "*_gt*_labelIds_19classes.png")

        self.imgs = list(sorted(glob.glob( imgs_path)))
        self.masks = list(sorted(glob.glob( masks_path)))

        troisdorf = root_imgs + "/leftImg8bit_trainextra/leftImg8bit/train_extra/troisdorf/troisdorf_000000_000073_leftImg8bit.png"
        if troisdorf in self.imgs:
            self.imgs.remove(troisdorf)

        self.transforms=transforms
        log.info("Dataset: Cityscape %s - %s images - %s masks",split,  len(self.imgs),len(self.masks))

if __name__ == "__main__":
    transforms = A.Compose([
        #A.RandomCrop(width=768, height=768),
        #A.RandomScale(scale_limit=(-0.5,1),always_apply=True,p=1.0),
        #A.Resize(p=1.0,width=1024, height=512),
        #A.RandomCrop(width=1024, height=512,always_apply=True,p=1.0),
        #A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],always_apply=True
        ),
        ToTensorV2()])

    cityscapesPath = "/home/l727r/Desktop/Cityscape"
    Cityscape_train = Cityscapes_coarse_dataset(cityscapesPath, "train", transforms=transforms)

    img, mask = Cityscape_train[2000]
    print(len(Cityscape_train))
    print(img.shape)
    print(torch.unique(mask))
    out = show_cityscape(img=img, mask=mask, alpha=0.9, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    out.show()






