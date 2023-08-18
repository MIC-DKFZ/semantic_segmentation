import random

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from lightning import LightningDataModule

from src.utils.utils import get_logger
from src.utils.config_utils import has_not_empty_attr

# set number of Threads to 0 for opencv and albumentations
cv2.setNumThreads(0)
# import logger
log = get_logger(__name__)


def seed_worker(worker_id):
    """
    from: https://github.com/MIC-DKFZ/image_classification/blob/master/base_model.py
        https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
        ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def inst_collate_fn(batch):
    """
    For using Instance Segmentation Datasets in DataLoader
    """
    return tuple(zip(*batch))


def get_max_steps(
    size_dataset,
    batch_size,
    num_devices,
    accumulate_grad_batches,
    num_epochs,
    drop_last=True,
) -> int:
    """
    Computing the number of  steps, needed for polynomial lr scheduler
    Considering the number of gpus and if accumulate_grad_batches is used

    Returns
    -------
    int:
        total number of steps
    int:
        number of steps per epoch
    """
    # How many steps per epoch in total
    if drop_last:
        steps_per_epoch = size_dataset // batch_size  # round off if drop_last=False
    else:
        steps_per_epoch = np.ceil(size_dataset / batch_size)  # round up if drop_last=False

    # For ddp and multiple gpus the effective batch sizes doubles
    steps_per_gpu = int(np.ceil(steps_per_epoch / num_devices))
    # Include accumulate_grad_batches
    steps_per_epoch = int(np.ceil(steps_per_gpu / accumulate_grad_batches))
    max_steps = num_epochs * steps_per_epoch

    return max_steps, steps_per_epoch


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        augmentations: DictConfig,
        instance_seg: bool = False,
    ) -> None:
        """
        __init__ the LightningModule
        save parameters

        Parameters
        ----------
        dataset : DictConfig
            config of the dataset, is called by hydra.src.instantiate(dataset,split=.., transforms=..)
        batch_size : int
            batch size for train dataloader
        val_batch_size : int
            batch size for val and test dataloader
        num_workers : int
            number of workers for all dataloaders
        augmentations : DictConfig
            config containing the augmentations for Train, Test and Validation
        """
        super().__init__()

        # parameters for dataloader
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        # data augmentations for train,val and test
        self.augmentations = augmentations
        # dataset which is defined in the config
        self.dataset = dataset

        if instance_seg:
            self.collate_fn = inst_collate_fn
        else:
            self.collate_fn = None

    def setup(self, stage: str = None) -> None:
        """
        Setting up the Datasets by initializing the augmentation and the dataloader

        Parameters
        ----------
        stage: str
            current stage which is given by Pytorch Lightning
        """
        # get the augmentations for train,validation and test set
        # define the datasets which are defined in the config
        # additional arguments are the split and the augmentations
        if stage in (None, "fit"):
            transforms_train = hydra.utils.instantiate(self.augmentations.train)
            self.DS_train = hydra.utils.instantiate(
                self.dataset, split="train", transforms=transforms_train
            )
        if stage in (None, "fit", "validate"):
            transforms_val = hydra.utils.instantiate(self.augmentations.val)
            self.DS_val = hydra.utils.instantiate(
                self.dataset, split="val", transforms=transforms_val
            )
        if stage in (None, "test"):
            if has_not_empty_attr(self.augmentations, "TEST"):
                transforms_test = hydra.utils.instantiate(self.augmentations.test)
            else:
                transforms_test = hydra.utils.instantiate(self.augmentations.val)
            self.DS_test = hydra.utils.instantiate(
                self.dataset, split="test", transforms=transforms_test
            )

    def max_steps(self) -> int:
        """
        Computing and Logging the number of training steps, needed for polynomial lr scheduler
        Considering the number of gpus and if accumulate_grad_batches is used

        Returns
        -------
        int:
            number of training steps
        """
        # computing the maximal number of steps for training
        max_steps, max_steps_epoch = get_max_steps(
            size_dataset=len(self.DS_train),
            batch_size=self.batch_size,
            num_devices=self.trainer.num_devices,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            num_epochs=self.trainer.max_epochs,
            drop_last=True,
        )
        # base_size = len(self.DS_train)
        # steps_per_epoch = base_size // self.batch_size
        # steps_per_gpu = int(np.ceil(steps_per_epoch / self.trainers.num_devices))
        # acc_steps_per_gpu = int(np.ceil(steps_per_gpu / self.trainers.accumulate_grad_batches))
        # max_steps = self.trainers.max_epochs * acc_steps_per_gpu

        log.info(
            "Number of Training steps: {}  ({} steps per epoch)".format(max_steps, max_steps_epoch)
        )

        max_steps_val, max_steps_epoch_val = get_max_steps(
            size_dataset=len(self.DS_val),
            batch_size=self.val_batch_size,
            num_devices=self.trainer.num_devices,
            accumulate_grad_batches=1,
            num_epochs=self.trainer.max_epochs,
            drop_last=False,
        )

        log.info(
            "Number of Validation steps: {}  ({} steps per epoch)".format(
                max_steps_val, max_steps_epoch_val
            )
        )
        return max_steps

    def train_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader :
            train dataloader
        """
        return DataLoader(
            self.DS_train,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader :
            validation dataloader
        """
        return DataLoader(
            self.DS_val,
            pin_memory=True,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
            # worker_init_fn=seed_worker,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader :
            test dataloader
        """
        return DataLoader(
            self.DS_test,
            pin_memory=True,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
            # worker_init_fn=seed_worker,
        )
