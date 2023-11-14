import argparse
import glob
import logging
import sys
from os.path import join
import os

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)

import hydra
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import cm

from src.utils.config_utils import build_predict_config
from src.utils.config_utils import get_CV_ensemble_config
from src.utils.utils import get_logger, set_lightning_logging
from src.visualization.visualizer import Visualizer

log = get_logger(__name__)
set_lightning_logging()


def show_prediction(overrides_cl: list, augmentation: str, split: str, axis: int) -> None:
    """
    Visualizing a Models Predictions
        Initializing the dataset defined in the config
        Initializing the model from the checkpoint(ckpt_dir)
        Enables to predict with a CV ensemble as well as a single model dependent on the cfg.ckpt_file
        Display img, prediction + mask using opencv

    Parameters
    ----------
    overrides_cl : list
        arguments from commandline to overwrite hydra config
    augmentation : str
        which augmentations to use (train,val,test or None)
    split : str
        which split of the dataset to use
    axis : int
        show img and gt side by side or on top of each other

    Requirements
    ----------
    cfg.ckpt_file: str
        a) path to a singleoutput folder
           e.g. .../logs/Cityscapes/hrnet_v1/run__batch_size_3/2023-08-18_10-13-05
        b) path to output folder of a cross validation, folder has to contain at least one fold_x
           e.g. .../logs/Cityscapes/hrnet_v1/run__batch_size_3/
    """
    # Init and Compose Hydra to get the config
    hydra.initialize(config_path="../config", version_base="1.3")
    cfg = hydra.compose(config_name="testing", overrides=overrides_cl)

    # Define Colormap
    color_map = "viridis"
    cmap = np.array(cm.get_cmap(color_map, cfg.dataset.num_classes).colors * 255, dtype=np.uint8)[
        :, 0:3
    ]

    # Check if cfg.ckpt_dir points to a single model or a CV folder
    ensemble_CV = any([x.startswith("fold_") for x in os.listdir(cfg.ckpt_dir)])

    # Load the config from the Checkpoint
    if ensemble_CV:
        # All runs inside CV have the same config (exept dataset.fold which is not relevant here)
        file = glob.glob(join(cfg.ckpt_dir, "fold_*", "*", ".hydra", "overrides.yaml"))[0]
    else:
        file = join(cfg.ckpt_dir, ".hydra", "overrides.yaml")
    cfg = build_predict_config(file, overrides)

    # Instantiating Model and load weights from a Checkpoint
    if ensemble_CV:
        cfg.model = get_CV_ensemble_config(cfg.ckpt_dir)
        model = hydra.utils.instantiate(cfg.trainermodule, cfg=cfg, _recursive_=False)
    else:
        ckpt_file = glob.glob(os.path.join(cfg.ckpt_dir, "checkpoints", "best_*"))[0]
        log.info("Checkpoint Directory: %s", ckpt_file)

        cfg.trainermodule._target_ += ".load_from_checkpoint"
        model = hydra.utils.instantiate(
            cfg.trainermodule, ckpt_file, strict=True, cfg=cfg, _recursive_=False
        )
    model = model.cuda()
    model.eval()

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
    # dataset = hydra.utils.instantiate(cfg.dataset.dataset, split=split, transforms=transforms)
    dataset = hydra.utils.instantiate(
        cfg.dataclass, split=split, transforms=transforms, _recursive_=False
    )
    # Check if data is normalized, if yes redo this during visualization of the image
    mean = None
    std = None
    for t in transforms.transforms:  # .transforms:
        if isinstance(t, A.Normalize):
            mean = t.mean
            std = t.std
            break

    # Create Visualizer Class
    visualizer = Visualizer(
        dataset,
        cmap,
        model,
        mean=mean,
        std=std,
        image_loader=cfg.img_loader,
        label_handler=cfg.label_handler,
        axis=axis,
    )

    # Create the cv2 Window
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1200, 1200)

    # Create Trackbar for Image Id and alpha value
    cv2.createTrackbar("Image ID", "Window", 0, len(dataset) - 1, visualizer.update_window)

    cv2.createTrackbar("alpha", "Window", 50, 100, visualizer.update_alpha)
    if cfg.dataset.segmentation_type == "semantic":
        cv2.createTrackbar("correctness", "Window", 0, 100, visualizer.update_alpha)

    # Load first image to get the number of channels
    img = dataset[0][0]
    if len(img.shape) == 2:
        channels = 2
    else:
        channels = img.shape[0]

    # Create the Trackbar for the Channel Parameter
    cv2.createTrackbar("Channel", "Window", -1, channels - 1, visualizer.update_channel_and_alpha)
    cv2.setTrackbarMin("Channel", "Window", -1)
    cv2.setTrackbarPos("Channel", "Window", -1)

    # Show the first image in window and start loop
    with torch.no_grad():
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

                print(
                    "Save"
                    f" {os.path.join('dataset_visualizations', 'pred_'+file_name + '__image.png')}"
                )
                print(
                    "Save"
                    f" {os.path.join('dataset_visualizations', 'pred_'+file_name + '__mask.png')}"
                )

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
        "--axis",
        type=int,
        default=1,
        help="1 for displaying images side by side, 0 for displaying images on top of each other",
    )
    args, overrides = parser.parse_known_args()
    augmentation = args.augmentation
    split = args.split
    axis = args.axis

    show_prediction(overrides_cl=overrides, augmentation=augmentation, split=split, axis=axis)
