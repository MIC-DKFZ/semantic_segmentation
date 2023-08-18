import argparse
import logging
import sys
import os
from os.path import join
import glob

logging.basicConfig(
    # level=logging.INFO,
    stream=sys.stdout,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)

import hydra
import torch
from torch.utils.data import DataLoader
import lightning as L
import numpy as np
import cv2

from src.utils.config_utils import has_not_empty_attr, get_CV_ensemble_config
from src.utils.utils import get_logger, set_lightning_logging
from src.callbacks.callbacks import SemSegPredictionWriter


log = get_logger(__name__)
set_lightning_logging()


class Prediction_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.img_files = glob.glob(join(root, "*"))
        self.transforms = transforms

    def load_img(self, idx: int) -> np.ndarray:
        """
        Load a single image (self.img_files[idx]).
        Output shape: [w,h,3]
        """
        # Read Image and Convert to RGB (cv2 reads images in BGR)
        img = cv2.imread(self.img_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def apply_transforms(self, img: np.ndarray) -> tuple:
        """
        Apply transforms to image and mask
        """
        if self.transforms:
            # Apply Albumentations Transforms if they are given
            transformed = self.transforms(image=img)
            img = transformed["image"]
        return img

    def __getitem__(self, idx):
        name = self.img_files[idx].rsplit("/")[-1].rsplit(".")[0]
        img = self.load_img(idx)
        img = self.apply_transforms(img)
        return img, name

    def __len__(self):
        return len(self.img_files)


def predict(
    input_dir: str, output_dir: str, overrides: list, save_probabilities: bool = False
) -> None:
    """
    Compose config from config/testing.yaml with overwrites from the checkpoint and the overwrites
    from commandline
    Instantiating Callbacks, Datamodule, Model and Trainer
    Enables to predict with a CV ensemble as well as a single model dependent on the cfg.ckpt_file
    Run the lightning prediction

    Parameters
    ----------
    input_dir: str
        path to a folder containing the files to be predicted
    output_dir: str
        path to a folder in which the predictions should be saved
    overrides: list
    save_probabilities: (optional) bool
        save a .npz containing the softmax probabilities for each file

    Requirements
    ----------
    cfg.ckpt_file: str
        a) path to a singleoutput folder
           e.g. .../logs/Cityscapes/hrnet_v1/run__batch_size_3/2023-08-18_10-13-05
        b) path to output folder of a cross validation, folder has to contain at least one fold_x
           e.g. .../logs/Cityscapes/hrnet_v1/run__batch_size_3/
    """
    # Instantiating Hydra and compose cfg
    hydra.initialize(config_path="config", version_base="1.3")
    cfg = hydra.compose(config_name="testing", overrides=overrides)

    # Check if cfg.ckpt_dir points to a single model or a CV folder
    ensemble_CV = any([x.startswith("fold_") for x in os.listdir(cfg.ckpt_dir)])

    # Load the config from the Checkpoint
    if ensemble_CV:
        # All runs inside CV have the same config (exept dataset.fold which is not relevant here)
        file = glob.glob(join(cfg.ckpt_dir, "fold_*", "*", "hydra", "overrides.yaml"))[0]
    else:
        file = join(cfg.ckpt_dir, "hydra", "overrides.yaml")
    cfg = build_predict_config(file, overrides)

    # Instantiating Model and load weights from a Checkpoint
    if ensemble_CV:
        cfg.model = get_CV_ensemble_config(cfg.ckpt_dir)
        model = hydra.utils.instantiate(cfg.trainermodule, model_config=cfg, _recursive_=False)
    else:
        ckpt_file = glob.glob(os.path.join(cfg.ckpt_dir, "checkpoints", "best_*"))[0]
        log.info("Checkpoint Directory: %s", ckpt_file)

        cfg.trainermodule._target_ += ".load_from_checkpoint"
        model = hydra.utils.instantiate(
            cfg.trainermodule, ckpt_file, strict=True, model_config=cfg, _recursive_=False
        )

    # Instantiating Augmentations
    transforms = hydra.utils.instantiate(cfg.augmentation.test)

    # Instantiating Dataset
    dataset = Prediction_Dataset(input_dir, transforms)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=cfg.val_batch_size, num_workers=8)

    # Instantiating Prediction Writer Callback
    callbacks = [SemSegPredictionWriter(output_dir, save_probabilities=save_probabilities)]
    # Instantiating remaining Callbacks
    for _, cb_conf in cfg.CALLBACKS.items():
        if cb_conf is not None:
            cb = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    # Instantiating trainer with trainer_args from config
    trainer_args = getattr(cfg, "pl_trainer") if has_not_empty_attr(cfg, "pl_trainer") else {}
    trainer = L.Trainer(callbacks=callbacks, logger=[], **trainer_args)

    # Start Predicting
    trainer.predict(model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Path to the folder which contains the images to be predicted"
    )
    parser.add_argument("-o", "--output", help="Path to the Output folder")
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help="Store Softmax probabilities",
    )

    args, overrides = parser.parse_known_args()
    predict(args.input, args.output, overrides, args.save_probabilities)
