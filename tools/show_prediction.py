import argparse
import os
import glob
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)

import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from matplotlib import cm

from trainers.Semantic_Segmentation_Trainer import SegModel
from trainers.Instance_Segmentation_Trainer import InstModel

from src.utils import has_not_empty_attr, get_logger
from datasets.DataModules import get_augmentations_from_config

log = get_logger(__name__)

from src.visualization import Visualizer


def show_prediction(
    overrides_cl: list, augmentation: str, split: str, segmentation: str, axis: int
) -> None:
    """
    Show Model Predictions
    Load Model and Dataset from the checkpoint(ckpt_dir)
    Show predictions of the model for the images in a small GUI

    Parameters
    ----------
    overrides_cl : list of strings
        arguments from the commandline to overwrite the config
    """
    # initialize hydra
    hydra.initialize(config_path="../config", version_base="1.1")

    # change working dir to checkpoint dir
    if os.getcwd().endswith("tools"):
        ORG_CWD = os.path.join(os.getcwd(), "..")
    else:
        ORG_CWD = os.getcwd()

    ckpt_dir = None
    for override in overrides_cl:
        if override.startswith("ckpt_dir"):
            ckpt_dir = override.split("=", 1)[1]
            break
    if ckpt_dir is None:
        log.error(
            "ckpt_dir has to be in th config. Run python show_prediction.py ckpt_dir=<some.path>"
        )
        quit()
    os.chdir(ckpt_dir)

    # load overrides from the experiment in the checkpoint dir
    overrides_ckpt = OmegaConf.load(os.path.join("hydra", "overrides.yaml"))
    # compose config by override with overrides_ckpt, afterwards override with overrides_cl
    cfg = hydra.compose(config_name="testing", overrides=overrides_ckpt + overrides_cl)

    # Get the TESTING.OVERRIDES to check if additional parameters should be changed
    if has_not_empty_attr(cfg, "TESTING"):
        if has_not_empty_attr(cfg.TESTING, "OVERRIDES"):
            overrides_test = cfg.TESTING.OVERRIDES
            # Compose config again with including the new overrides
            cfg = hydra.compose(
                config_name="testing",
                overrides=overrides_ckpt + overrides_test + overrides_cl,
            )

    # load the best checkpoint and load the model
    cfg.ORG_CWD = ORG_CWD
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]
    # if hasattr(cfg.MODEL, "PRETRAINED"):
    #    cfg.MODEL.PRETRAINED = False
    if segmentation == "semantic":
        model = SegModel.load_from_checkpoint(ckpt_file, model_config=cfg, strict=False).cuda()
    elif segmentation == "instance":
        model = InstModel.load_from_checkpoint(ckpt_file, model_config=cfg, strict=False).cuda()
    # model = SegModel.load_from_checkpoint(ckpt_file, config=cfg).cuda()
    # print(cfg)
    # print(cfg.model)

    # model=hydra.utils.instantiate(cfg.model).cuda()
    OmegaConf.set_struct(cfg, False)
    if augmentation == "train":
        transforms = get_augmentations_from_config(cfg.AUGMENTATIONS.TRAIN)[0]
    elif augmentation == "val":
        transforms = get_augmentations_from_config(cfg.AUGMENTATIONS.VALIDATION)[0]
    elif augmentation == "test":
        transforms = get_augmentations_from_config(cfg.AUGMENTATIONS.TEST)[0]
    else:
        transforms = A.Compose([ToTensorV2()])

    # instantiate dataset
    dataset = hydra.utils.instantiate(cfg.dataset, split=split, transforms=transforms)

    # check if data is normalized, if yes redo this during visualization of the image
    mean = None
    std = None
    for t in transforms.transforms:  # .transforms:
        if isinstance(t, A.Normalize):
            mean = t.mean
            std = t.std
            break

    # define colormap
    color_map = "viridis"
    cmap = np.array(cm.get_cmap(color_map, cfg.DATASET.NUM_CLASSES).colors * 255, dtype=np.uint8)[
        :, 0:3
    ]

    # init visualizer
    visualizer = Visualizer(
        dataset, cmap, model, mean=mean, std=std, segmentation=segmentation, axis=axis
    )

    # create window
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1200, 1200)

    # Create Trackbar for Image Id and alpha value
    cv2.createTrackbar("Image ID", "Window", 0, len(dataset) - 1, visualizer.update_window)

    cv2.createTrackbar("alpha", "Window", 50, 100, visualizer.update_alpha)
    if segmentation == "semantic":
        cv2.createTrackbar("correctness", "Window", 0, 100, visualizer.update_alpha)

    # look at the first image to get the number of channels
    img, _ = dataset[0]
    if len(img.shape) == 2:
        channels = 2
    else:
        channels = img.shape[0]

    # Create the Trackbar for the Channel Parameter
    cv2.createTrackbar("Channel", "Window", -1, channels - 1, visualizer.update_channel_and_alpha)
    cv2.setTrackbarMin("Channel", "Window", -1)
    cv2.setTrackbarPos("Channel", "Window", -1)

    # show the first image in window and start loop
    model.eval()
    with torch.no_grad():
        visualizer.update_window()
        print("press q to quit")
        while True:
            k = cv2.waitKey()
            if k == 113:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation",
        type=str,
        default="test",
        help="Which augmentations to use: train, val or test (by default)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="which split to use: train, val or test (by default)",
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

    show_prediction(overrides, augmentation, split, segmentation, axis)
