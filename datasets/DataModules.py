import hydra
from omegaconf import OmegaConf, DictConfig
import numpy as np

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

import albumentations as A
import albumentations.pytorch
import utils.augmentations as custom_augmentations
import cv2

from utils.utils import has_not_empty_attr
from utils.utils import get_logger

# set number of Threads to 0 for opencv and albumentations
cv2.setNumThreads(0)
# import logger
log = get_logger(__name__)


def get_max_steps(
    size_dataset, batch_size, num_devices, accumulate_grad_batches, num_epochs, drop_last=True
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


def get_augmentations_from_config(augmentations: DictConfig) -> list:
    """
    Build an Albumentations augmentation pipeline from the input config

    Parameters
    ----------
    augmentations : DictConfig
        config of the Augmentation

    Returns
    -------
    list :
        list of Albumentations transforms
    """
    # print(augmentations)
    # print(type(augmentations))
    # if transformations are given as a Albumentations-dict they can directly be loaded
    if has_not_empty_attr(augmentations, "FROM_DICT"):
        return [A.from_dict(OmegaConf.to_container(augmentations.FROM_DICT))]

    # otherwise recursively build the transformations
    trans = []
    for augmentation in augmentations:

        transforms = list(augmentation.keys())

        for transform in transforms:
            parameters = getattr(augmentation, transform)
            if parameters is None:
                parameters = {}

            if hasattr(A, transform):
                if "transforms" in list(parameters.keys()):
                    # "transforms" indicates a transformation which takes a list of other transformations
                    # as input ,e.g. A.Compose -> recursively build these transforms
                    transforms = get_augmentations_from_config(parameters.transforms)
                    del parameters["transforms"]
                    func = getattr(A, transform)
                    trans.append(func(transforms=transforms, **parameters))
                else:
                    # load transformation form Albumentations
                    func = getattr(A, transform)
                    trans.append(func(**parameters))
            elif hasattr(A.pytorch, transform):
                # ToTensorV2 transformation is located in A.pytorch
                func = getattr(A.pytorch, transform)
                trans.append(func(**parameters))
            elif hasattr(custom_augmentations, transform):
                func = getattr(custom_augmentations, transform)
                trans.append(func(**parameters))
            else:
                log.info("No Operation Found: %s", transform)
    return trans


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        augmentations: DictConfig,
    ) -> None:
        """
        __init__ the LightningModule
        save parameters

        Parameters
        ----------
        dataset : DictConfig
            config of the dataset, is called by hydra.utils.instantiate(dataset,split=.., transforms=..)
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
            transforms_train = get_augmentations_from_config(self.augmentations.TRAIN)[0]
            # print(transforms_train)
            self.DS_train = hydra.utils.instantiate(
                self.dataset, split="train", transforms=transforms_train
            )
        if stage in (None, "fit", "validate"):
            transforms_val = get_augmentations_from_config(self.augmentations.VALIDATION)[0]
            self.DS_val = hydra.utils.instantiate(
                self.dataset, split="val", transforms=transforms_val
            )
        if stage in (None, "test"):
            if has_not_empty_attr(self.augmentations, "TEST"):
                transforms_test = get_augmentations_from_config(self.augmentations.TEST)[0]
            else:
                transforms_test = get_augmentations_from_config(self.augmentations.VALIDATION)[0]
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
        # steps_per_gpu = int(np.ceil(steps_per_epoch / self.trainer.num_devices))
        # acc_steps_per_gpu = int(np.ceil(steps_per_gpu / self.trainer.accumulate_grad_batches))
        # max_steps = self.trainer.max_epochs * acc_steps_per_gpu

        log.info("Number of Training steps: {}".format(max_steps))
        log.info("Number of Training steps per epoch: {}".format(max_steps_epoch))

        max_steps_val, max_steps_epoch_val = get_max_steps(
            size_dataset=len(self.DS_val),
            batch_size=self.val_batch_size,
            num_devices=self.trainer.num_devices,
            accumulate_grad_batches=1,
            num_epochs=self.trainer.max_epochs,
            drop_last=False,
        )

        log.info("Number of Validation steps: {}".format(max_steps_val))
        log.info("Number of Validation steps per epoch: {}".format(max_steps_epoch_val))
        return max_steps

    def get_augmentations_from_config(self, augmentations: DictConfig) -> list:
        """
        Build an Albumentations augmentation pipeline from the input config

        Parameters
        ----------
        augmentations : DictConfig
            config of the Augmentation

        Returns
        -------
        list :
            list of Albumentations transforms
        """
        # if transformations are given as a Albumentations-dict they can directly be loaded
        if has_not_empty_attr(augmentations, "FROM_DICT"):
            return [A.from_dict(OmegaConf.to_container(augmentations.FROM_DICT))]

        # otherwise recursively build the transformations
        trans = []
        for augmentation in augmentations:

            transforms = list(augmentation.keys())

            for transform in transforms:
                parameters = getattr(augmentation, transform)
                if parameters is None:
                    parameters = {}

                if hasattr(A, transform):
                    if "transforms" in list(parameters.keys()):
                        # "transforms" indicates a transformation which takes a list of other transformations
                        # as input ,e.g. A.Compose -> recursively build these transforms
                        transforms = self.get_augmentations_from_config(parameters.transforms)
                        del parameters["transforms"]
                        func = getattr(A, transform)
                        trans.append(func(transforms=transforms, **parameters))
                    else:
                        # load transformation form Albumentations
                        func = getattr(A, transform)
                        trans.append(func(**parameters))
                elif hasattr(A.pytorch, transform):
                    # ToTensorV2 transformation is located in A.pytorch
                    func = getattr(A.pytorch, transform)
                    trans.append(func(**parameters))
                else:
                    log.info("No Operation Found: %s", transform)
        return trans

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
        )
