import os
import glob
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.datasets.Cityscapes.Cityscapes import Cityscapes_dataset
from src.utils.utils import get_logger

log = get_logger(__name__)


# dataset class for using fine and coarse cityscapes data
# subclass of the cityscapes dataset and just adopt the init
# coarse_portion: defines the amount of coarse data which should be included, between 0 (none) and 1 (all)
class Cityscape_fine_coarse_dataset(Cityscapes_dataset):
    def __init__(self, root, split="train", transforms=None, coarse_portion=1.0):
        # providing the possibility to have data and labels at different locations
        if isinstance(root, str):
            root_imgs = root
            root_labels = root
        else:
            root_imgs = root.IMAGES
            root_labels = root.LABELS

        # no test dataset for cityscapes so return the validation set instead
        if split == "test":
            split = "val"

        if split == "train":
            # building the paths for fine and coarse images
            imgs_path_fine = os.path.join(
                root_imgs,
                "leftImg8bit_trainvaltest",
                "leftImg8bit",
                split,
                "*",
                "*_leftImg8bit.png",
            )
            imgs_path_coarse = os.path.join(
                root_imgs,
                "leftImg8bit_trainextra",
                "leftImg8bit",
                "train_extra",
                "*",
                "*_leftImg8bit.png",
            )

            # building the paths for fine and coarse masks
            masks_path_fine = os.path.join(
                root_labels,
                "gtFine_trainvaltest",
                "gtFine",
                split,
                "*",
                "*_gt*_labelIds_19classes.png",
            )
            masks_path_coarse = os.path.join(
                root_labels,
                "gtCoarse",
                "gtCoarse",
                "train_extra",
                "*",
                "*_gt*_labelIds_19classes.png",
            )
            # elif num_classes==34:
            #    masks_path_fine = os.path.join(root_labels, "gtFine_trainvaltest", "gtFine", split, "*","*_gt*_labelIds.png")
            #    masks_path_coarse = os.path.join(root_labels, "gtCoarse", "gtCoarse", "train_extra", "*","*_gt*_labelIds.png")

            # save all path in lists
            imgs_fine = list(sorted(glob.glob(imgs_path_fine)))
            imgs_coarse = list(sorted(glob.glob(imgs_path_coarse)))

            # this image is corrupt, so exclude it
            troisdorf = (
                root_imgs
                + "/leftImg8bit_trainextra/leftImg8bit/train_extra/troisdorf/troisdorf_000000_000073_leftImg8bit.png"
            )
            if troisdorf in imgs_coarse:
                imgs_coarse.remove(troisdorf)
            masks_fine = list(sorted(glob.glob(masks_path_fine)))
            masks_coarse = list(sorted(glob.glob(masks_path_coarse)))

            # randomly select coarse_portion of the coarse data
            coarse_portion = max(coarse_portion, 0)
            indices = random.sample(range(len(imgs_coarse)), int(len(imgs_coarse) * coarse_portion))
            indices.sort()
            imgs_coarse = [imgs_coarse[index] for index in indices]
            masks_coarse = [masks_coarse[index] for index in indices]

            # join file and selected coarse data
            self.masks = masks_fine + masks_coarse
            self.imgs = imgs_fine + imgs_coarse

            log.info(
                "Dataset: Cityscape %s (Coarse+Fine) | Total: %s images - %s masks | Fine: %s"
                " images - %s masks | Fine: %s images - %s masks",
                split,
                len(self.imgs),
                len(self.masks),
                len(imgs_fine),
                len(masks_fine),
                len(imgs_coarse),
                len(masks_coarse),
            )

        elif split == "val":
            # for validation only the fine annotated dat is used
            # building the paths
            imgs_path = os.path.join(
                root_imgs,
                "leftImg8bit_trainvaltest",
                "leftImg8bit",
                split,
                "*",
                "*_leftImg8bit.png",
            )
            masks_path = os.path.join(
                root_labels,
                "gtFine_trainvaltest",
                "gtFine",
                split,
                "*",
                "*_gt*_labelIds_19classes.png",
            )

            # save all path in lists
            self.imgs = list(sorted(glob.glob(imgs_path)))
            self.masks = list(sorted(glob.glob(masks_path)))

            log.info(
                "Dataset: Cityscape %s (Coarse+Fine) | Total: %s images - %s masks",
                split,
                len(self.imgs),
                len(self.masks),
            )

        self.transforms = transforms


if __name__ == "__main__":
    transforms = A.Compose(
        [
            # A.RandomCrop(width=768, height=768),
            # A.RandomScale(scale_limit=(-0.5,1),always_apply=True,p=1.0),
            # A.Resize(p=1.0,width=1024, height=512),
            # A.RandomCrop(width=1024, height=512,always_apply=True,p=1.0),
            # A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),
        ]
    )

    cityscapesPath = "/home/l727r/Desktop/Cityscape"
    Cityscape_train = Cityscape_fine_coarse_dataset(
        cityscapesPath, "train", transforms=transforms, coarse_portion=-0.2
    )
    # for i in range(0,50):
    img, mask = Cityscape_train[2000]
