import os
import glob
from collections import namedtuple
import logging
log = logging.getLogger(__name__)

import torch
import torchvision.utils

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import random
from datasets.Cityscapes import Cityscapes_dataset,show_cityscape

class Cityscape_fine_coarse_dataset(Cityscapes_dataset):
    def __init__(self,root,split="train",transforms=None,coarse_portion=1.0):

        if isinstance(root, str):
            root_imgs=root
            root_labels=root
        else:
            root_imgs = root.IMAGES
            root_labels = root.LABELS

        if split=="test":
            split="val"

        if split=="train":
            imgs_path_fine=os.path.join( root_imgs,"leftImg8bit_trainvaltest", "leftImg8bit" , split , "*" , "*_leftImg8bit.png" )
            imgs_path_coarse=os.path.join( root_imgs ,"leftImg8bit_trainextra", "leftImg8bit" , "train_extra" , "*" , "*_leftImg8bit.png" )

            #if num_classes==19:
            masks_path_fine = os.path.join(root_labels, "gtFine_trainvaltest", "gtFine", split, "*", "*_gt*_labelIds_19classes.png")
            masks_path_coarse = os.path.join(root_labels, "gtCoarse", "gtCoarse", "train_extra", "*", "*_gt*_labelIds_19classes.png")
            #elif num_classes==34:
            #    masks_path_fine = os.path.join(root_labels, "gtFine_trainvaltest", "gtFine", split, "*","*_gt*_labelIds.png")
            #    masks_path_coarse = os.path.join(root_labels, "gtCoarse", "gtCoarse", "train_extra", "*","*_gt*_labelIds.png")


            imgs_fine = list(sorted(glob.glob( imgs_path_fine)))
            imgs_coarse = list(sorted(glob.glob( imgs_path_coarse)))

            troisdorf=root_imgs+"/leftImg8bit_trainextra/leftImg8bit/train_extra/troisdorf/troisdorf_000000_000073_leftImg8bit.png"
            if troisdorf in imgs_coarse:
                imgs_coarse.remove(troisdorf)
            masks_fine = list(sorted(glob.glob( masks_path_fine)))
            masks_coarse = list(sorted(glob.glob( masks_path_coarse)))

            coarse_portion=max(coarse_portion,0)
            indices = random.sample(range(len(imgs_coarse)), int(len(imgs_coarse)*coarse_portion))
            indices.sort()

            imgs_coarse=[imgs_coarse[index] for index in indices]
            masks_coarse=[masks_coarse[index] for index in indices]
            self.masks=masks_fine+masks_coarse
            self.imgs=imgs_fine+imgs_coarse

            log.info(
                "Dataset: Cityscape %s (Coarse+Fine) | Total: %s images - %s masks | Fine: %s images - %s masks | Fine: %s images - %s masks",
                split, len(self.imgs), len(self.masks), len(imgs_fine), len(masks_fine), len(imgs_coarse),
                len(masks_coarse))


        elif split == "val":
            imgs_path = os.path.join(root_imgs, "leftImg8bit_trainvaltest", "leftImg8bit", split, "*",
                                     "*_leftImg8bit.png")
            masks_path = os.path.join(root_labels, "gtFine_trainvaltest", "gtFine", split, "*",
                                      "*_gt*_labelIds_19classes.png")

            self.imgs = list(sorted(glob.glob(imgs_path)))
            self.masks = list(sorted(glob.glob(masks_path)))

            log.info(
                "Dataset: Cityscape %s (Coarse+Fine) | Total: %s images - %s masks",
                split, len(self.imgs), len(self.masks))

        self.transforms=transforms


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
    Cityscape_train = Cityscape_coarse_dataset(cityscapesPath, "train", transforms=transforms,coarse_portion=-0.2)
    #for i in range(0,50):
    img, mask = Cityscape_train[2000]
    print(len(Cityscape_train))
    print(img.shape)
    print(torch.unique(mask))
    out = show_cityscape(img=img, mask=mask, alpha=0.9, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    out.show()
    #out.save("out.png")





