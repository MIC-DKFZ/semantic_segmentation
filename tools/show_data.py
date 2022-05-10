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
from matplotlib import cm

from utils.utils import get_logger
from tools.show_prediction import Visualizer

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


def show_data(overrides_cl: list) -> None:
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
    hydra.initialize(config_path="../config")
    cfg = hydra.compose(config_name="baseline", overrides=overrides_cl)

    # Define Colormap and basic Transforms and instantiate the dataset
    color_map = "viridis"
    cmap = np.array(cm.get_cmap(color_map, cfg.DATASET.NUM_CLASSES).colors * 255, dtype=np.uint8)[
        :, 0:3
    ]
    transforms = A.Compose([ToTensorV2()])
    dataset = hydra.utils.instantiate(cfg.dataset, split="train", transforms=transforms)

    visualizer = Visualizer(dataset, cmap)

    # Create the cv2 Window
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1200, 1200)

    # Create Trackbar for Image Id and alpha value
    cv2.createTrackbar("Image ID", "Window", 0, len(dataset) - 1, visualizer.update_window)
    cv2.createTrackbar("alpha", "Window", 0, 100, visualizer.update_alpha)

    # look at the first image to get the number of channels
    img, _ = dataset[0]
    if len(img.shape) == 2:
        channels = 2
    else:
        channels = img.shape[0]
    # Create the Trackbar for the Channel
    cv2.createTrackbar("Channel", "Window", -1, channels - 1, visualizer.update_channel_and_alpha)
    cv2.setTrackbarMin("Channel", "Window", -1)
    cv2.setTrackbarPos("Channel", "Window", -1)

    # show the first image in window and start loop
    visualizer.update_window()
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
