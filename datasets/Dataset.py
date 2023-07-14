import os
from os.path import join, split
import glob
from collections import namedtuple
import json
import torch
import torchvision.utils
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils import get_logger

log = get_logger(__name__)
from typing import Tuple
from numpy import ndarray
from src.dataset_utils import random_crop, center_crop, split_by_ID

cv2.setNumThreads(0)

# Needed Information:
# 1. k-fold Cross Validation Splits
# 2. For each Class: Which Images contain the Class
# 3. For each Image: For each Class: Sampling Points

# nnUNet: 1000 epochs with 250 batches
class Base_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        img_folder: str,
        mask_folder: str,
        num_classes: int,
        fold: int = 0,
        dtype: str = ".png",
        split: str = "train",
        transforms=None,
        patch_size: tuple = (512, 512),
        num_points_per_class: int = 10000,
        steps_per_epoch: int = 100,  # 250*batch_size
        class_probabilities: list = None,
        ignore_bg: bool = False,
        random_sampling: float = 0.3,
        img_prefix: str = "",
        img_postfix: str = "",
        mask_prefix: str = "",
        mask_postfix: str = "",
    ):
        # TODO - validation behaviour
        # TODO - Add class_probabilities properly
        # TODO - When there is not a single file which contains the class the sampling fails
        # TODO - Padding if needed, in the crop functions or in the transforms
        # TODO - Maybe Augbemtation for the Class Sampled Crop?
        self.root = root
        self.split = split
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.dtype = dtype
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.random_sampling = random_sampling
        self.class_probabilities = class_probabilities
        self.steps_per_epoch = steps_per_epoch * 250
        self.num_points_per_class = num_points_per_class

        self.img_prefix = img_prefix
        self.img_postfix = img_postfix
        self.mask_prefix = mask_prefix
        self.mask_postfix = mask_postfix

        # Get the image and mask files
        self.img_files = self.get_imgs()
        self.mask_files = self.get_masks()
        print(f"Dataset: total - {len(self.img_files)} Images and {len(self.mask_files)} Masks")

        self.preprocessing()

        # Load the splits_final.json to define which images should be used for this fold and split
        with open(join(self.root, "splits_final.json")) as splits_final:
            splits_final = json.load(splits_final)[fold][split]

        # selected_elements=
        mask_files_filtered = []
        img_files_filtered = []

        for mask_file, img_file in zip(self.mask_files, self.img_files):
            if any(s in img_file for s in splits_final):
                mask_files_filtered.append(mask_file)
                img_files_filtered.append(img_file)

        self.mask_files = mask_files_filtered
        self.img_files = img_files_filtered

        self.transforms = transforms
        print(f"Dataset: {split} - {len(self.img_files)} Images and {len(self.img_files)} Masks")

    def get_imgs(self):
        img_files = glob.glob(
            join(
                self.root,
                self.img_folder,
                self.img_prefix + "*" + self.img_postfix + self.dtype,
            )
        )
        img_files = list(sorted(img_files))

        return img_files

    def get_masks(self):
        mask_files = glob.glob(
            join(
                self.root,
                self.mask_folder,
                self.mask_prefix + "*" + self.mask_postfix + self.dtype,
            )
        )

        mask_files = list(sorted(mask_files))
        return mask_files

    def preprocessing(self):

        # Define the sampling points for each file in the dataset
        if not os.path.exists(join(self.root, "class_occurrences.json")):

            # For each mask check which labels occur and save them
            os.makedirs(join(self.root, "class_locations"), exist_ok=True)
            class_occurence = [[] for _ in range(self.num_classes)]
            for mask_file in self.mask_files:
                mask_name = split(mask_file)[-1].replace(self.dtype, "")

                sample_points, classes = self.define_sampling_points(mask_file)

                # Save a Dict for each file which contains sampling locations for each class inside the image
                with open(join(self.root, "class_locations", mask_name + ".json"), "w") as file:
                    json.dump(sample_points, file)  # {Class-ID: {'x': x, 'y': y}}
                for c in classes:
                    # if c >= 0 and c < self.num_classes:
                    class_occurence[c].append(mask_name)

            # Summary of which classes occur in which files
            with open(join(self.root, "class_occurrences.json"), "w") as file:
                json.dump(class_occurence, file)  # {Class-ID: {'x': x, 'y': y}}

        # Define the splits for Training and Validation if "splits_final.json" not already exists
        if not os.path.exists(join(self.root, "splits_final.json")):
            splits = self.define_splits(self.img_files)
            with open(join(self.root, "splits_final.json"), "w") as file:
                json.dump(splits, file, sort_keys=True, indent=4)  # [{'train':[],'val':[]}]

    def define_splits(self, files):
        print("Splits")
        # files = [file.split(join(self.root, self.img_folder))[-1] for file in files]
        files = [os.path.relpath(file, join(self.root, self.img_folder)) for file in files]
        files = [file.rsplit("/", 1)[-1] for file in files]
        splits = split_by_ID(files)
        return splits

        # pass

    def get_stats(self):
        print("Stats")
        pass

    def define_sampling_points(self, file):
        sample_points = {}

        # Load the Mask
        mask = cv2.imread(file, -1)

        # Check which Class-IDs exist in the Mask an iterate over them
        unique_vals = np.unique(mask)
        for unique_val in unique_vals:
            # Catch Ignore Label
            if unique_val < 0 or unique_val >= self.num_classes:
                continue

            x, y = np.where(mask == unique_val)
            num_pixels = len(x)
            # select a defined number of pixels (x,y) in which have the Class-ID
            idx = np.random.choice(
                np.arange(num_pixels),
                min(num_pixels, self.num_points_per_class),
                replace=False,
            )
            x = x[idx]
            y = y[idx]

            # Add them to the Dict
            sample_points[str(unique_val)] = {"x": x.tolist(), "y": y.tolist()}

        return sample_points, unique_vals

    def __getitem__(self, idx):

        if self.split == "train":
            return self.get_training_sample(idx)
        else:
            return self.get_validation_sample(idx)

    def get_validation_sample(self, idx):
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_files[idx], -1)

        # apply albumentations transforms
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"].long()
        return img, mask

    def get_training_sample(self, idx):
        if np.random.random() <= self.random_sampling:
            # Randomly Select an Image
            idx = np.random.randint(0, len(self.img_files))
            # Load Image and Mask
            img = cv2.imread(self.img_files[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order
            mask = cv2.imread(self.mask_files[idx], -1)
            # Randomly Crop Image and Mask
            img, mask = random_crop(img, mask, self.patch_size)

        else:
            # Randomly select a class using the given class_probabilities
            class_id = np.random.choice(
                np.arange(0, self.num_classes), p=self.class_probabilities
            )  # p=probs / sum(probs))

            # Randomly select an Image which contains the Class
            with open(join(self.root, "class_occurrences.json"), "r") as file:
                data = json.load(file)[class_id]
                data = [
                    mask for mask in data if any(mask in mask_file for mask_file in self.mask_files)
                ]
                img_file = np.random.choice(data)

            # Randomly choose a sampling point form the list
            with open(join(self.root, "class_locations", img_file + ".json"), "r") as file:
                data = json.load(file)[str(class_id)]
                pt_idx = np.random.randint(0, len(data["x"]))
                x, y = data["x"][pt_idx], data["y"][pt_idx]

            # Find the index of the selected file in self.mask_files
            idx = [i for i, s in enumerate(self.mask_files) if img_file in s][0]
            # Load Image and Mask
            img = cv2.imread(self.img_files[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order
            mask = cv2.imread(self.mask_files[idx], -1)

            # Crop Image and Mask
            img, mask = center_crop(img, mask, x, y, self.patch_size)

        # Apply albumentations transforms if they are defined
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"].long()

        return img, mask

    def __len__(self):

        if self.split == "train":
            return self.steps_per_epoch
        else:
            return len(self.img_files)


if __name__ == "__main__":
    path = "/media/l727r/data/nnUNet/nnUNetv2_raw/Dataset253_COMPUTING_it3"
    Dataset = Base_Dataset(
        root=path,
        img_folder="imagesTr",
        mask_folder="labelsTr",
        split="val",
        num_classes=8,
        random_sampling=0.3,
    )
    img, mask = Dataset[0]
    print(img.shape, mask.shape)
