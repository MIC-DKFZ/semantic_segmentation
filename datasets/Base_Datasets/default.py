import glob
import os
from os.path import join
from typing import Any

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils import get_logger

log = get_logger(__name__)


class Base_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        img_folder: str,
        label_folder: str,
        split: str = "train",
        transforms: Any = None,
        dtype: str = ".png",
        img_folder_val: str = None,
        label_folder_val: str = None,
        img_folder_test: str = None,
        label_folder_test: str = None,
        img_prefix: str = "",
        img_postfix: str = "",
        label_prefix: str = "",
        label_postfix: str = "",
    ):
        # Dataset Properties
        self.root = root
        self.split = split
        self.transforms = transforms

        # Manage Folder Structure to have more flexibility for organizing train,val and test Data
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.img_folder_val = img_folder_val if img_folder_val else img_folder
        self.label_folder_val = label_folder_val if label_folder_val else label_folder
        self.img_folder_test = img_folder_test if img_folder_test else img_folder_val
        self.label_folder_test = label_folder_test if label_folder_test else label_folder_val

        # Manage File Naming Structure to have more flexibility
        self.img_prefix = img_prefix
        self.img_postfix = img_postfix
        self.label_prefix = label_prefix
        self.label_postfix = label_postfix
        self.dtype = dtype

        # Load all Path for Image and Mask Files
        self.img_files = self.get_img_files()
        self.mask_files = self.get_mask_files()

        log.info(
            f"Dataset {self.split}: {len(self.img_files)} images - {len(self.mask_files)} masks"
        )
        print(f"Dataset {self.split}: {len(self.img_files)} images - {len(self.mask_files)} masks")

    def get_img_files(self) -> list:
        folder = self.img_folder
        folder = self.img_folder_val if self.split == "val" else folder
        folder = self.img_folder_test if self.split == "test" else folder

        img_files = glob.glob(
            join(
                self.root,
                folder,
                self.img_prefix + "*" + self.img_postfix + self.dtype,
            )
        )
        img_files = list(sorted(img_files))
        return img_files

    def get_mask_files(self) -> list:
        folder = self.label_folder
        folder = self.label_folder_val if self.split == "val" else folder
        folder = self.label_folder_test if self.split == "test" else folder

        mask_files = glob.glob(
            join(
                self.root,
                folder,
                self.label_prefix + "*" + self.label_postfix + self.dtype,
            )
        )
        mask_files = list(sorted(mask_files))

        return mask_files

    def load_img(self, idx):
        # Read Image and Convert to RGB (cv2 reads images in BGR)
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_mask(self, idx):
        return cv2.imread(self.mask_files[idx], -1)

    def load_data(self, idx: int) -> tuple:
        # Load Image and Mask
        img = self.load_img(idx)
        mask = self.load_mask(idx)
        return img, mask

    def apply_transforms(self, img: np.ndarray, mask: np.ndarray) -> tuple:
        # Apply Albumentations Transforms if they are given
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"].long()
        return img, mask

    def __getitem__(self, idx: int) -> tuple:
        # Load Image and Mask
        img, mask = self.load_data(idx)
        # Apply Data Augmentations
        img, mask = self.apply_transforms(img, mask)

        return img, mask

    def __len__(self) -> int:
        return len(self.img_files)


if __name__ == "__main__":

    # Define some Transformations
    transforms = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),
        ]
    )

    # Create A Dataset
    Cityscape_DS = Base_Dataset(
        root="../../../Datasets/cityscapes",
        img_folder="leftImg8bit_trainvaltest/leftImg8bit/train/*/",
        img_folder_val="leftImg8bit_trainvaltest/leftImg8bit/val/*/",
        label_folder="gtFine_trainvaltest/gtFine/train/*/",
        label_folder_val="gtFine_trainvaltest/gtFine/val/*/",
        label_postfix="labelIds_19classes",
        split="train",
        transforms=transforms,
    )

    # Load Some Data
    img, mask = Cityscape_DS[100]
    print(img.shape, mask.shape)
