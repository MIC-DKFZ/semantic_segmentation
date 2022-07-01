import os
import glob
import time

import torch
import numpy as np
import cv2
from numcodecs import Blosc, blosc
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import pickle as pkl
from utils.visualization_utils import show_data
from utils.utils import get_logger
import zarr
from tqdm import tqdm
import json
from numcodecs import blosc

blosc.use_threads = False

log = get_logger(__name__)

ignore_label = 255


class AGGC2022_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms, fold="None"):
        self.imgs = glob.glob(os.path.join(root, "imgs", "*png"))
        self.masks = glob.glob(os.path.join(root, "masks", "*png"))

        self.imgs.sort()
        self.masks.sort()
        # get your data for the corresponding split
        len_train = int(len(self.imgs) * 0.8)
        if split == "train":
            self.imgs = self.imgs[0:len_train]
            self.masks = self.masks[0:len_train]
        if (
            split == "val" or split == "test"
        ):  # if you have dont have a test set use the validation set
            self.imgs = self.imgs[len_train:-1]
            self.masks = self.masks[len_train:-1]
        # save the transformations
        self.transforms = transforms
        log.info(
            "Dataset: AGGC2022 %s - Fold %s - %s images - %s masks",
            split,
            fold,
            len(self.imgs),
            len(self.masks),
        )

    def __getitem__(self, idx):
        # reading images and masks as numpy arrays
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order

        mask = cv2.imread(self.masks[idx], -1)
        # thats how you apply Albumentations transformations
        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


class AGGC2022_tiff_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        transforms,
        **kwargs,
    ):
        self.root = root
        self.imgs = glob.glob(os.path.join(root, "Subset*", "imgs", "*.zarr"))
        self.imgs = [img for img in self.imgs if "Subset3" not in img]
        self.masks = glob.glob(os.path.join(root, "Subset*", "masks", "*.zarr"))
        print("{} Images and {} Masks found".format(len(self.imgs), len(self.masks)))
        self.transforms = transforms
        # len_train = int(len(self.imgs) * 0.8)
        # if split == "train":
        #    self.imgs = self.imgs[0:len_train]
        # if (
        #    split == "val" or split == "test"
        # ):  # if you have dont have a test set use the validation set
        #    self.imgs = self.imgs[len_train:-1]
        # self.transforms = transforms
        # self.root = root
        # self.path_imgs = "/media/l727r/data/AGGC2022/Subset1/imgs_zarr/Subset1_Train_1.zarr"
        # self.path_bb = "/media/l727r/data/AGGC2022/Subset1/boxes/Subset1_Train_1.json"
        # self.path_masks = "/media/l727r/data/AGGC2022/Subset1/masks_zarr/Subset1_Train_1.zarr"

    def __getitem__(self, idx):

        # open zarr image
        # img = zarr.open(self.imgs[idx], mode="r")
        # mask = zarr.open(self.mask[idx], mode="r")
        print(self.imgs[idx])
        img = zarr.open(self.imgs[idx], mode="r")
        mask = zarr.open(self.masks[idx], mode="r")

        # path_bb = self.masks[idx].replace("masks", "boxes").replace(".zarr", ".json")
        # with open(path_bb) as f:
        #    bbs = json.load(f)
        # boxes = list(bbs.values())

        path_sp = self.masks[idx].replace("masks", "sample_points").replace(".zarr", ".pkl")
        with open(path_sp, "rb") as f:
            sps = pkl.load(f)
        points = list(sps.values())
        print(sps.keys())
        index_c = np.random.randint(0, len(points))
        index_b = np.random.randint(0, len(points[index_c]))
        print(np.array(points).shape, len(points), len(points[index_c]))  # , points)
        print(points[index_c][0][index_b], points[index_c][1][index_b])
        x = points[index_c][0][index_b]
        y = points[index_c][1][index_b]
        # x_min, y_min, x_max, y_max = boxes[index_c][index_b]
        # img = img[x_min:x_max, y_min:y_max, :]
        # mask = mask[x_min:x_max, y_min:y_max]
        ps_half = 1024
        img = img[x - ps_half : x + ps_half, y - ps_half : y + ps_half, :]
        mask = mask[x - ps_half : x + ps_half, y - ps_half : y + ps_half]
        # albumentation transforms
        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]
        # print(img.shape, mask.shape)
        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataset = AGGC2022_tiff_dataset("/media/l727r/data/AGGC2022/Subset1")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=6,
        num_workers=12,
        drop_last=True,
        persistent_workers=True,
    )
    start = time.time()
    for x in tqdm(dataloader):
        s = x
    end = time.time()
    print("Time: {}".format(end - start))
