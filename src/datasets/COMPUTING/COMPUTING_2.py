import glob

import cv2
import torch
import json
import os
from src.utils.utils import get_logger

log = get_logger(__name__)

from typing import Tuple
from numpy import ndarray


def random_crop(
    img: ndarray,
    mask: ndarray,
    patch_size: Tuple = (600, 600),
):
    """
    Crop the Image around the center (x,y) with a random shift in x and y direction
    in range [-(patch_size*offset) : (patch_size*offset))

    Parameters
    ----------
    img: ndarray
        input image, should be RGB with shape (with,height,3)
    patch_size: Tuple, optional
        Size of the output patch

    Returns
    -------
    ndarray
        cropped image with patch_size
    """
    w, h, _ = img.shape

    # Get the random offset in x and y direction defined by offset parameter
    x = np.random.randint(0, w - patch_size[0])
    y = np.random.randint(0, h - patch_size[1])

    # Clip the patch to be inside the image with min=0 and max=(w,h) - patch_size
    img_cropped = img[x : x + patch_size[0], y : y + patch_size[1], :]
    mask_cropped = mask[x : x + patch_size[0], y : y + patch_size[1]]

    # Zero padding if img shape is smaller than patch size, only occurs when patch_size > img_size
    # if img_cropped.shape != (patch_size[0], patch_size[1], 3):
    #     w, h, _ = img_cropped.shape
    #     dif_x = patch_size[0] - w
    #     dif_y = patch_size[1] - h
    #     img_cropped = np.pad(
    #         img_cropped,
    #         (
    #             (int(np.trunc(dif_x / 2)), int(np.ceil(dif_x / 2))),
    #             (int(np.trunc(dif_y / 2)), int(np.ceil(dif_y / 2))),
    #             (0, 0),
    #         ),
    #     )

    return img_cropped, mask_cropped


def random_crop_around_center(
    img: ndarray,
    mask: ndarray,
    x: int,
    y: int,
    patch_size: Tuple = (600, 600),
    offset: float = 0.5,
):
    """
    Crop the Image around the center (x,y) with a random shift in x and y direction
    in range [-(patch_size*offset) : (patch_size*offset))

    Parameters
    ----------
    img: ndarray
        input image, should be RGB with shape (with,height,3)
    x: int
        y coordinate of the center point around the patch is cropped
    y: int
        y coordinate of the center point around the patch is cropped
    patch_size: Tuple, optional
        Size of the output patch
    offset: float, optional
        ratio of the patch size which defines the range of the shift in x and y direction

    Returns
    -------
    ndarray
        cropped image with patch_size
    """
    w, h, c = img.shape

    # Get the random offset in x and y direction defined by offset parameter
    offset_range_x = int(patch_size[0] * offset)
    offset_range_y = int(patch_size[1] * offset)
    offset_x = np.random.randint(-offset_range_x, offset_range_x)
    offset_y = np.random.randint(-offset_range_y, offset_range_y)

    # Shift center point to top-left corner
    # Adding the random offset
    # Clip the patch to be inside the image with min=0 and max=(w,h) - patch_size
    x = int(max(min(x - np.ceil(patch_size[0] / 2) + offset_x, w - patch_size[0]), 0))
    y = int(max(min(y - np.ceil(patch_size[1] / 2) + offset_y, h - patch_size[1]), 0))

    # Copping the image
    img_cropped = img[x : x + patch_size[0], y : y + patch_size[1], :]
    mask_cropped = mask[x : x + patch_size[0], y : y + patch_size[1]]

    # Zero padding if img shape is smaller than patch size, only occurs when patch_size > img_size
    # if img_cropped.shape != (patch_size[0], patch_size[1], 3):
    #     w, h, _ = img_cropped.shape
    #     dif_x = patch_size[0] - w
    #     dif_y = patch_size[1] - h
    #     img_cropped = np.pad(
    #         img_cropped,
    #         (
    #             (int(np.trunc(dif_x / 2)), int(np.ceil(dif_x / 2))),
    #             (int(np.trunc(dif_y / 2)), int(np.ceil(dif_y / 2))),
    #             (0, 0),
    #         ),
    #     )
    #

    return img_cropped, mask_cropped


class Computing_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        transforms,
        fold=0,
        num_classes=9,
        patch_size=(600, 600),
        num_points_per_class=2000,
        steps_per_epoch=100,
        class_probabilities=None,
        random_sampling=0.3,
    ):
        # get your data for the corresponding split
        self.__getitem__ = self.__getitem__train
        if split == "test":
            split = "val"
        self.split = split

        with open(os.path.join(root, "splits_final.json")) as f:
            splits = json.load(f)[fold][split]

        self.imgs = [os.path.join(root, "imagesTr", name + "_0000.png") for name in splits]
        self.masks = [os.path.join(root, "labelsTr", name + ".png") for name in splits]

        # save the transformations
        self.transforms = transforms
        self.num_classes = num_classes
        self.steps_per_epoch = steps_per_epoch
        self.random_sampling = random_sampling
        self.patch_size = patch_size
        self.class_probabilities = class_probabilities

        self.sample_points = [[]] * num_classes
        for file in self.masks:
            mask = cv2.imread(file, -1)
            name = file.rsplit("/", 1)[1]
            unique_vals = np.unique(mask)
            for unique_val in unique_vals:
                x, y = np.where(mask == unique_val)
                num_pixels = len(x)
                idx = np.random.choice(
                    np.arange(num_pixels), min(num_pixels, num_points_per_class), replace=False
                )
                x = x[idx]
                y = y[idx]
                self.sample_points[unique_val].append(
                    {"file": name, "x": x, "y": y, "p": num_pixels}
                )

    def __getitem__(self, idx):
        if self.split == "train":
            return self.__getitem__train(idx)
        else:
            return self.__getitem__val(idx)

    def __getitem__train(self, idx):

        if np.random.random() <= self.random_sampling:
            idx = np.random.randint(0, len(self.imgs))
            img = cv2.imread(self.imgs[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order
            mask = cv2.imread(self.masks[idx], -1)
            img, mask = random_crop(img, mask, self.patch_size)

        else:
            # Randomly select a class and an image which contains the img
            selecte_class = np.random.randint(0, self.num_classes - 1)
            probs = np.array([x["p"] for x in self.sample_points[selecte_class]])
            sample = np.random.choice(self.sample_points[selecte_class], p=probs / sum(probs))

            # Randomly select a Pixel of the Class
            point_id = np.random.randint(0, len(sample["x"]))
            x = sample["x"][point_id]
            y = sample["y"][point_id]

            idx = [i for i, file in enumerate(self.masks) if sample["file"] in file][0]

            # reading images and masks as numpy arrays
            img = cv2.imread(self.imgs[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order
            mask = cv2.imread(self.masks[idx], -1)

            # Crop the image and mask to conatain the select point
            img, mask = random_crop_around_center(img, mask, x, y, self.patch_size)

        # that's how you apply Albumentations transformations
        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        return img, mask.long()

    def __getitem__val(self, idx):
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
        if self.split == "train":
            return self.steps_per_epoch  # len(self.imgs)
        else:
            return len(self.imgs)


if __name__ == "__main__":
    from os.path import join
    import numpy as np

    root = "/media/l727r/data/nnUNet/nnUNetv2_raw/Dataset251_COMPUTING_it1"
    num_classes = 9
    num_points_per_class = 10

    sample_points = [[]] * num_classes

    labesl_path = join(root, "labelsTr")

    files = glob.glob(join(labesl_path, "*.png"))
    for file in files:
        sample_points_file = {}
        mask = cv2.imread(file, -1)
        name = file.rsplit("/", 1)[1]
        unique_vals = np.unique(mask)
        for unique_val in unique_vals:
            x, y = np.where(mask == unique_val)
            num_pixels = len(x)
            idx = np.random.choice(
                np.arange(num_pixels), min(num_pixels, num_points_per_class), replace=False
            )
            x = x[idx]
            y = y[idx]
            sample_points[unique_val].append({"file": name, "x": x, "y": y, "p": num_pixels})
            # sample_points_file[str(unique_val)]= [x,y]
    print(sample_points[1])
    probs = np.array([x["p"] for x in sample_points[0]])
    sample = np.random.choice(sample_points[1], p=probs / sum(probs))
    point_id = np.random.randint(0, len(sample["x"]))
    x = sample["x"][point_id]
    y = sample["y"][point_id]
    print(sample["file"], x, y)
    img = cv2.imread(join(root, "imagesTr", sample["file"].replace(".png", "_0000.png")))
    mask = cv2.imread(join(root, "labelsTr", sample["file"]))
    print(img.shape)
    img, mask = random_crop_around_center(img, mask, x, y)
    print(img.shape)

    cv2.imshow("W", img)
    cv2.imshow("B", mask * 255)
    cv2.waitKey()
    # print(np.unique(mask))

    # print(join(labesl_path, "*"))
    # print(files)
