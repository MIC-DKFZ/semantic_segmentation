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
from datasets.Base_Datasets.sampling import Sampling_Dataset
from src.utils import get_logger
from src.dataset_utils import random_crop, center_crop  # , split_by_ID

log = get_logger(__name__)


def split_by_ID(files, ids=None, num_folds=5):
    if ids is None:
        ids = files
    splits = []
    for i in range(0, num_folds):
        val_ids = ids[i::num_folds]
        train_ids = [ID for ID in ids if ID not in val_ids]

        val_samples = [file for file in files if any(val_id in file for val_id in val_ids)]
        train_samples = [file for file in files if any(train_id in file for train_id in train_ids)]
        print(f"Split {i}, #Train: {len(train_samples)}, #Val: {len(val_samples)}")
        splits.append({"train": train_samples, "val": val_samples})
    return splits


class CV_Dataset(Base_Dataset):
    def __init__(self, fold=0, num_folds=5, split_ids=None, **kwargs):
        self.fold = fold
        self.num_folds = num_folds
        self.split_ids = split_ids

        super().__init__(**kwargs)

        self.preprocessing_splitting()

        with open(join(self.root, "splits_final.json")) as splits_file:
            splits_final = json.load(splits_file)[self.fold][self.split]

        mask_files_filtered = []
        img_files_filtered = []

        # Just take the image and mask pairs inside the splits_final
        for mask_file, img_file in zip(self.mask_files, self.img_files):
            if any(s in img_file for s in splits_final):
                mask_files_filtered.append(mask_file)
                img_files_filtered.append(img_file)

        self.mask_files = mask_files_filtered
        self.img_files = img_files_filtered

        print(
            f"Dataset {self.split} - {len(self.img_files)} Images and {len(self.img_files)} Masks"
        )

    def preprocessing_splitting(self):
        if os.path.exists(join(self.root, "splits_final.json")):
            return
        log.info(f"Split Data into {self.num_folds} Folds")
        print(f"Split Data into {self.num_folds} Folds")

        files = [split(file)[-1] for file in self.img_files]
        splits = split_by_ID(files, ids=self.split_ids, num_folds=self.num_folds)

        with open(join(self.root, "splits_final.json"), "w") as file:
            json.dump(splits, file, indent=4)  # [{'train':[],'val':[]}]


class CV_Sampling_Dataset(CV_Dataset, Sampling_Dataset):
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
    ids = [
        "aachen",
        "cologne",
        "erfurt",
        "jena",
        "strasbourg",
        "ulm",
        "bochum",
        "darmstadt",
        "hamburg",
        "krefeld",
        "stuttgart",
        "weimar",
        "bremen",
        "dusseldorf",
        "hanover",
        "monchengladbach",
        "tubingen",
        "zurich",
    ]
    # Create A Dataset
    Cityscape_DS = CV_Dataset(
        root="../../../Datasets/cityscapes",
        img_folder="leftImg8bit_trainvaltest/leftImg8bit/train/*/",
        img_folder_val="leftImg8bit_trainvaltest/leftImg8bit/val/*/",
        label_folder="gtFine_trainvaltest/gtFine/train/*/",
        label_folder_val="gtFine_trainvaltest/gtFine/val/*/",
        label_postfix="labelIds_19classes",
        split="train",
        transforms=transforms,
        split_ids=ids,
    )

    # Load Some Data
    img, mask = Cityscape_DS[100]
    print(img.shape, mask.shape)

    # Create A Dataset
    Cityscape_DS = CV_Sampling_Dataset(
        root="../../../Datasets/cityscapes",
        img_folder="leftImg8bit_trainvaltest/leftImg8bit/train/*/",
        img_folder_val="leftImg8bit_trainvaltest/leftImg8bit/val/*/",
        label_folder="gtFine_trainvaltest/gtFine/train/*/",
        label_folder_val="gtFine_trainvaltest/gtFine/val/*/",
        label_postfix="labelIds_19classes",
        split="train",
        transforms=transforms,
        split_ids=ids,
        num_classes=19,
    )

    # Load Some Data
    img, mask = Cityscape_DS[100]
    print(img.shape, mask.shape)
