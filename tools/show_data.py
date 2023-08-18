import argparse
import logging
import os
import sys

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)

import hydra

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import cm

from src.utils.utils import get_logger, set_lightning_logging
from src.utils.visualization import Visualizer

log = get_logger(__name__)
set_lightning_logging()


def show_data(
    overrides_cl: list, augmentation: str, split: str, segmentation: str, axis: int
) -> None:
    """
    Visualizing a Dataset
        Initializing the dataset defined in the config
        Display img + mask using opencv

    Parameters
    ----------
    overrides_cl : list
        arguments from commandline to overwrite hydra config
    augmentation : str
        which augmentations to use (train,val,test or None)
    split : str
        which split of the dataset to use
    segmentation : str
        which type of segmentation is used (semantic, instance or multilabel)
    axis : int
        show img and gt side by side or on top of each other
    """
    # Init and Compose Hydra to get the config
    hydra.initialize(config_path="../config", version_base="1.3")
    cfg = hydra.compose(config_name="training", overrides=overrides_cl)

    # Define Colormap
    color_map = "viridis"
    cmap = np.array(cm.get_cmap(color_map, cfg.DATASET.NUM_CLASSES).colors * 255, dtype=np.uint8)[
        :, 0:3
    ]

    # Instantiating Augmentations
    if augmentation is None:
        transforms = A.Compose([ToTensorV2()])
    elif augmentation == "train":
        transforms = hydra.utils.instantiate(cfg.augmentation.train)
    elif augmentation == "val":
        transforms = hydra.utils.instantiate(cfg.augmentation.val)
    elif augmentation == "test":
        transforms = hydra.utils.instantiate(cfg.augmentation.test)

    # Instantiating Dataset
    dataset = hydra.utils.instantiate(cfg.dataset, split=split, transforms=transforms)

    # Check if data is normalized, if yes redo this during visualization of the image
    mean = None
    std = None
    for t in transforms.transforms:  # .transforms:
        if isinstance(t, A.Normalize):
            mean = t.mean
            std = t.std
            break

    # Create Visualizer Class
    visualizer = Visualizer(dataset, cmap, mean=mean, std=std, segmentation=segmentation, axis=axis)

    # Create the cv2 Window
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1200, 1200)

    # Create Trackbar for Image Id and alpha value
    cv2.createTrackbar("Image ID", "Window", 0, len(dataset) - 1, visualizer.update_window)
    cv2.createTrackbar("alpha", "Window", 0, 100, visualizer.update_alpha)

    # Load first image to get the number of channels
    img = dataset[0][0]
    if len(img.shape) == 2:
        channels = 2
    else:
        channels = img.shape[0]
    # Create the Trackbar for the Channel
    cv2.createTrackbar("Channel", "Window", -1, channels - 1, visualizer.update_channel_and_alpha)
    cv2.setTrackbarMin("Channel", "Window", -1)
    cv2.setTrackbarPos("Channel", "Window", -1)

    # Show the first image in window and start loop
    visualizer.update_window()
    print("press q to quit")
    while True:
        print("Press q to quit \n Press s to save the current image and mask")
        k = cv2.waitKey()
        if k == 113:
            break
        elif k == 115:

            img_id = cv2.getTrackbarPos("Image ID", "Window")
            file_name = f"{cfg.DATASET.NAME}__ID{img_id}"
            os.makedirs("dataset_visualizations", exist_ok=True)

            print(f"Save {file_name}")

            img = cv2.cvtColor(visualizer.img_np_fig, cv2.COLOR_RGB2BGR)
            mask = cv2.cvtColor(visualizer.mask_np, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join("dataset_visualizations", file_name + "__image.png"), img)
            cv2.imwrite(os.path.join("dataset_visualizations", file_name + "__mask.png"), mask)

    cv2.destroyAllWindows()


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
        help="semantic, instance or multilabel, depending on the dataset",
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
