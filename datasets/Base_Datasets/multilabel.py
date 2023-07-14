import glob
from os.path import join, split
import os
from typing import Any
import json

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.Base_Datasets.default import Base_Dataset
from datasets.Base_Datasets.cross_validation import CV_Dataset
from src.utils import get_logger
from src.dataset_utils import random_crop, center_crop, split_by_ID

log = get_logger(__name__)


class Multilabel_Dataset(Base_Dataset):
    def apply_transforms(self, img: np.ndarray, mask: np.ndarray) -> tuple:
        if self.transforms is not None:
            mask = mask.transpose((1, 2, 0))
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1).float().long()
        return img, mask


class Multilabel_CV_Dataset(Multilabel_Dataset, CV_Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == "__main__":

    # Define some Transformations
    transforms = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),
        ]
    )

    # Create A Dataset
    Fishinspector_DS = Base_Dataset(
        root="/media/l727r/data/UFZ_2023_Fishinspector/Dataset222_Fishinspector",
        img_folder="imagesTr",
        label_folder="imagesTr",
        split="train",
        transforms=transforms,
    )

    # Load Some Data
    img, mask = Fishinspector_DS[100]
    print(img.shape, mask.shape)
