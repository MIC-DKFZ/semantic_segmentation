import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)


import hydra
from omegaconf import OmegaConf

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


from utils.utils import get_logger
from matplotlib import cm

log = get_logger(__name__)

# OmegaConf resolver for preventing problems in the output path
OmegaConf.register_new_resolver(
    "path_formatter",
    lambda s: s.replace("[", "")
    .replace("]", "")
    .replace("}", "")
    .replace("{", "")
    .replace(")", "")
    .replace("(", "")
    .replace(",", "_")
    .replace("=", "_")
    .replace("/", ".")
    .replace("+", ""),
)


def show_data(overrides) -> None:
    """
    Visualizing a Dataset
    initializing the dataset defined in the config
    display img + mask using opencv

    Parameters
    ----------
    overrides : dict
        arguments from commandline to overwrite hydra config
    """

    def update_window(val):
        # Get parameters from Trackbars
        alpha = cv2.getTrackbarPos("alpha", "Window") / 100
        img_id = cv2.getTrackbarPos("Image ID", "Window")
        channel_id = cv2.getTrackbarPos("Channel", "Window")

        # Load Image and Mask and transform to numpy array
        img, mask = dataset[img_id]
        img = np.array(img)
        mask = np.array(mask)

        # if img has only one channel transform to 3 channels, otherwise change the axis
        # from c,w,h to w,h,c
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = np.moveaxis(img, 0, -1)

        # only use the selected channel, if -1 than use the first 3 channels
        if channel_id == -1:
            img = img[:, :, 0:3]
        else:
            img = img[:, :, channel_id]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # colormap the mask
        w, h = mask.shape
        fig = np.zeros((w, h, 3), dtype=np.uint8)
        for class_id in np.unique(mask):
            x, y = np.where(mask == class_id)
            if class_id > len(cmap):
                fig[x, y] = [0, 0, 0]
            else:
                fig[x, y, :] = cmap[class_id]

        # blend the image and the mask, dependent on the alpha value
        img = cv2.addWeighted(img, 1 - alpha, fig, alpha, 0.0)
        # concat blended image and mask
        fig = np.concatenate((img, fig), 1)
        # transform from RGB to BGR to match the cv2 order
        fig = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)
        # show image
        cv2.imshow("Window", fig)

    # Init and Compose Hydra to get the config
    hydra.initialize(config_path="../config")
    cfg = hydra.compose(config_name="baseline", overrides=overrides)

    # Define Colormap and basic Transforms and instantiate the dataset
    color_map = "viridis"
    cmap = np.array(cm.get_cmap(color_map, cfg.DATASET.NUM_CLASSES).colors * 255, dtype=np.uint8)[
        :, 0:3
    ]
    transforms = A.Compose([ToTensorV2()])
    dataset = hydra.utils.instantiate(cfg.dataset, split="train", transforms=transforms)

    # Create the cv2 Window
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1200, 1200)

    # Create Trackbar for Image Id and alpha value
    cv2.createTrackbar("Image ID", "Window", 0, len(dataset) - 1, update_window)
    cv2.createTrackbar("alpha", "Window", 0, 100, update_window)

    # look at the first image to get the number of channels
    img, _ = dataset[0]
    if len(img.shape) == 2:
        channels = 2
    else:
        channels = img.shape[0]
    # Create the Trackbar for the Channel
    cv2.createTrackbar("Channel", "Window", -1, channels - 1, update_window)
    cv2.setTrackbarMin("Channel", "Window", -1)
    cv2.setTrackbarPos("Channel", "Window", -1)

    # show the first image in window and start loop
    update_window(0)
    print("press q to quit")
    while True:
        k = cv2.waitKey()
        if k == 113:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args, overrides = parser.parse_known_args()

    show_data(overrides)
