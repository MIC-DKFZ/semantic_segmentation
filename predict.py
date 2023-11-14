from typing import List
import argparse
import logging
import sys
import os
from os.path import join, split
import glob

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)

import numpy as np
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning import LightningModule, Callback, Trainer
import albumentations

from src.callbacks.prediction_writer import PredictionWriter

from src.utils.config_utils import has_not_empty_attr, get_CV_ensemble_config, build_predict_config
from src.utils.utils import get_logger, set_lightning_logging

log = get_logger(__name__)
set_lightning_logging()

# LightningModule.load_from_checkpoint()


class Prediction_Dataset(Dataset):
    def __init__(self, root: str, img_loader, transforms=None, dtype=".png"):

        self.transforms = transforms
        self.dtype = dtype
        self.img_loader = hydra.utils.instantiate(img_loader)
        self.img_files = glob.glob(join(root, f"*{self.dtype}"))

    def apply_transforms(self, img: np.ndarray) -> tuple:
        """
        Apply transforms to image and mask
        """
        if self.transforms:
            # Apply Albumentations Transforms if they are given
            transformed = self.transforms(image=img)
            img = transformed["image"]
        return img

    def __getitem__(self, idx: int):
        name = split(self.img_files[idx])[-1].replace(self.dtype, "")
        img = self.img_loader.load_img(self.img_files[idx])
        img = self.apply_transforms(img)
        return img, name

    def __len__(self):
        return len(self.img_files)


def predict(
    input_dir: str,
    output_dir: str,
    overrides: list,
    ckpt_type: str = "best",
    save_probabilities: bool = False,
    save_visualsation: bool = False,
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
    # Config - Instantiating Hydra and compose cfg
    hydra.initialize(config_path="config", version_base="1.3")
    cfg: DictConfig = hydra.compose(config_name="testing", overrides=overrides)

    # Config - Check if cfg.ckpt_dir points to a single model or a CV folder
    ensemble_CV: bool = any([x.startswith("fold_") for x in os.listdir(cfg.ckpt_dir)])

    # Config - Load the config from the Checkpoint
    if ensemble_CV:
        # All runs inside CV have the same config (exept dataset.fold which is not relevant here)
        file: str = glob.glob(join(cfg.ckpt_dir, "fold_*", "*", ".hydra", "overrides.yaml"))[0]
    else:
        file: str = join(cfg.ckpt_dir, ".hydra", "overrides.yaml")
    cfg: DictConfig = build_predict_config(file, overrides)

    # Model - Instantiating and load weights from a Checkpoint
    if ensemble_CV:
        cfg.model.model: DictConfig = get_CV_ensemble_config(cfg.ckpt_dir, ckpt_type=ckpt_type)
        model: LightningModule = hydra.utils.instantiate(
            cfg.trainermodule, cfg=cfg, _recursive_=False
        )
    else:
        ckpt_file = glob.glob(os.path.join(cfg.ckpt_dir, "checkpoints", ckpt_type + "_*"))[0]
        log.info("Checkpoint Directory: %s", ckpt_file)

        cfg.trainermodule._target_ += ".load_from_checkpoint"
        model = hydra.utils.instantiate(
            cfg.trainermodule, ckpt_file, strict=True, cfg=cfg, _recursive_=False
        )

    # Augmentations - Instantiating Albumentations Augmentations
    transforms: albumentations.augmentations.transforms = hydra.utils.instantiate(
        cfg.augmentation.test
    )
    # Dataset - Instantiating
    dataset: Dataset = Prediction_Dataset(input_dir, cfg.img_loader, transforms, cfg.dataset.dtype)
    dataloader: DataLoader = DataLoader(
        dataset, shuffle=False, batch_size=cfg.val_batch_size, num_workers=cfg.num_workers
    )

    callbacks = [
        PredictionWriter(output_dir, cfg.label_handler, save_probabilities, save_visualsation)
    ]

    # Callback - Instantiating remaining Callbacks
    for _, cb_conf in cfg.callbacks.items():
        if cb_conf is not None:
            cb: Callback = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb)

    # Trainer - Instantiating with trainer_args from config (cfg.pl_trainer)
    trainer_args: dict = getattr(cfg, "pl_trainer") if has_not_empty_attr(cfg, "pl_trainer") else {}
    trainer: Trainer = L.Trainer(callbacks=callbacks, logger=[], **trainer_args)

    # Predicting
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
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Store visualization",
    )
    parser.add_argument("-c", "--ckpt_type", default="best", help="best or last")

    args, overrides = parser.parse_known_args()
    predict(
        args.input,
        args.output,
        overrides,
        args.ckpt_type,
        args.save_probabilities,
        args.save_visualization,
    )
