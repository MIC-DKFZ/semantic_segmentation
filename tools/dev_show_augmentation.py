import argparse
import logging
import os
import sys

from src.utils.visualization import convert_numpy_to

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)

import hydra
from omegaconf import OmegaConf
import albumentations as A
import cv2
import numpy as np
from matplotlib import cm

from src.utils.utils import get_logger

log = get_logger(__name__)


def show_data(
    overrides_cl: list, augmentation: str, split: str, segmentation: str, axis: int
) -> None:
    """
    Visualizing a Dataset
    initializing the dataset defined in the config
    display img + mask using opencv

    Parameters
    ----------
    overrides_cl : list
        arguments from commandline to overwrite hydra config
    """
    # Init and Compose Hydra to get the config
    hydra.initialize(config_path="../config", version_base="1.1")
    cfg = hydra.compose(config_name="baseline", overrides=overrides_cl)

    # Define Colormap and basic Transforms and instantiate the dataset
    color_map = "viridis"
    cmap = np.array(cm.get_cmap(color_map, cfg.DATASET.NUM_CLASSES).colors * 255, dtype=np.uint8)[
        :, 0:3
    ]

    OmegaConf.set_struct(cfg, False)
    # transforms = A.Compose([])
    transforms = A.Compose([])
    # if augmentation is None:
    #     transforms = A.Compose([ToTensorV2()])
    # elif augmentation == "train":
    #     transforms = get_augmentations_from_config(cfg.AUGMENTATIONS.TRAIN)[0]
    # elif augmentation == "val":
    #     transforms = get_augmentations_from_config(cfg.AUGMENTATIONS.VALIDATION)[0]
    # elif augmentation == "test":
    #     transforms = get_augmentations_from_config(cfg.AUGMENTATIONS.TEST)[0]

    dataset = hydra.utils.instantiate(cfg.dataset, split=split, transforms=transforms)

    img, mask = dataset[0]
    print(img.shape, type(img))
    # img = img.permute((1, 2, 0))#.float()  # .to(torch.uint8)
    num_bins = 10
    bins = np.linspace(0.1, 1.0, num_bins)
    total = img
    for bin in bins:
        # print(bin)
        aug = A.ColorJitter(brightness=(bin, bin), contrast=0, saturation=0, hue=0, p=1)
        img_tmp = aug(image=img)["image"]
        print(type(img_tmp), img_tmp.dtype, np.min(img_tmp), np.max(img_tmp))
        img_tmp = cv2.putText(
            img_tmp,
            f"{bin:.4f}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            [255, 0, 0],
            2,
            cv2.LINE_AA,
        )
        total = np.concatenate((total, img_tmp), 1)

    # convert_torch_to(img.permute((1, 2, 0)), "PIL").show()

    convert_numpy_to(total, "PIL").show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        help="Which augmentations to use: None (by default), train, val or test",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="which split to use: train (by default), val or test",
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        default="semantic",
        help="semantic or instance, depending on the dataset",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=1,
        help="1 for displaying images side by side, 0 for displaying images on top of each other",
    )
    args, overrides = parser.parse_known_args()
    augmentation = args.augmentation
    split = args.split
    segmentation = args.segmentation
    axis = args.axis

    show_data(overrides, augmentation, split, segmentation, axis)
