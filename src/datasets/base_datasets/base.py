import glob
from os.path import join
from typing import Any, List, Tuple, Dict, Optional

import hydra.utils
from omegaconf.dictconfig import DictConfig
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import augmentations


from src.utils.utils import get_logger

cv2.setNumThreads(0)
log = get_logger(__name__)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        img_loader,
        label_handler,
        img_folder: str,
        label_folder: str,
        img_folder_val: Optional[str] = None,
        label_folder_val: Optional[str] = None,
        img_folder_test: Optional[str] = None,
        label_folder_test: Optional[str] = None,
        dtype: str = ".png",
        dtype_mask: str = ".png",
        img_prefix: str = "",
        img_postfix: str = "",
        label_prefix: str = "",
        label_postfix: str = "",
        split: str = "train",
        transforms: augmentations.transforms = None,
        *args: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Base Dataset for loading images and mask by:
            1) __init__: Collect Image and Mask files (train-val split)
            2) __getitem__: Load Images + apply Augmentations

        Parameters
        ----------
        root: str
            Root to the data
        img_folder: str
            Folder which contains the images, also folder structures work (e.g. images/*/train/)
        label_folder: str
            Folder which contains the labels, also folder structures work (e.g. labels/*/train/)
        split: str, optional
            Train, val or test set
        transforms:
            Albumentations transformation which are applied to the image and mask pairs
        dtype: str, optional
            Datatype of the Image Files
        dtype_mask: str, optional
            Datatype of the Mask Files
        img_folder_val: Optional[str], optional
            Folder which contains the (val)images, also folder structures work (e.g. images/*/val/)
        label_folder_val: Optional[str], optional
            Folder which contains the (val)labels, also folder structures work (e.g. labels/*/val/)
        img_folder_test: Optional[str], optional
            Folder which contains the (test)images, also folder structures work (e.g. images/*/test/)
        label_folder_test: Optional[str], optional
            Folder which contains the (test)labels, also folder structures work (e.g. labels/*/test/)
        img_prefix: str, optional
            Prefix of the image files
        img_postfix: str, optional
            Postfix of the image files
        label_prefix: str, optional
            Prefix of the label files
        label_postfix: str, optional
             Postfix of the label files
        args
        kwargs
        """

        # Dataset Properties
        self.root = root
        self.split = split
        self.transforms = transforms

        # Manage Folder Structure to have more flexibility for organizing train,val and test Data
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.img_folder_val = img_folder_val  # if img_folder_val else img_folder
        self.label_folder_val = label_folder_val  # if label_folder_val else label_folder
        self.img_folder_test = img_folder_test
        self.label_folder_test = label_folder_test

        if not img_folder_test or not label_folder_test:
            log.info("No Test set is defined: Validation set is used instead during testing")
            self.img_folder_test = img_folder_val
            self.label_folder_test = label_folder_val
        else:
            self.img_folder_test = img_folder_test
            self.label_folder_test = label_folder_test

        # Manage File Naming Structure to have more flexibility
        self.img_prefix = img_prefix
        self.img_postfix = img_postfix
        self.label_prefix = label_prefix
        self.label_postfix = label_postfix
        self.dtype = dtype
        self.dtype_mask = dtype_mask

        # Setup Data Handlers
        self.img_handler = (
            hydra.utils.instantiate(img_loader)
            if isinstance(img_loader, DictConfig)
            else img_loader
        )
        self.label_handler = (
            hydra.utils.instantiate(label_handler)
            if isinstance(label_handler, DictConfig)
            else label_handler
        )

        self.preprocessing()
        # Setup Data Files
        self.setup_files()

        log.info(
            f"Dataset {self.split}: {len(self.img_files)} images - {len(self.mask_files)} masks"
        )
        # print(f"Dataset {self.split}: {len(self.img_files)} images - {len(self.mask_files)} masks")

    def preprocessing(self) -> None:
        pass

    def setup_files(self):
        """
        Get Paths to the Image and Mask Files
        """
        if self.split == "test" and self.img_folder_test is None:
            log.info("No Test set is defined: Validation set is used instead during testing")
            split = "val"
        else:
            split = self.split

        self.img_files: List[str] = self.get_img_files(split)
        self.mask_files: List[str] = self.get_mask_files(split)

    def get_img_files(self, split) -> List[str]:
        """
        Return the path to the images in the current split.
        Outputs of get_img_files and get_mask_files have to be in the same order.

        Returns
        -------
        List[str]
            List of absolute path to the images in the current split
        """
        folder = self.img_folder
        folder = self.img_folder_val if split == "val" else folder
        folder = self.img_folder_test if split == "test" else folder

        img_files = glob.glob(
            join(self.root, folder, f"{self.img_prefix}*{self.img_postfix}{self.dtype}")
        )

        img_files = list(sorted(img_files))
        return img_files

    def get_mask_files(self, split) -> List[str]:
        """
        Return the path to the masks in the current split.
        Outputs of get_img_files and get_mask_files have to be in the same order.

        Returns
        -------
        List[str]
            List of absolute path to the masks in the current split
        """
        folder = self.label_folder
        folder = self.label_folder_val if split == "val" else folder
        folder = self.label_folder_test if split == "test" else folder

        mask_files = glob.glob(
            join(self.root, folder, f"{self.label_prefix}*{self.label_postfix}{self.dtype_mask}")
        )

        mask_files = list(sorted(mask_files))
        return mask_files

    def load_data(self, idx):
        img = self.img_handler.load_img(self.img_files[idx])
        mask = self.label_handler.load_mask(self.mask_files[idx])
        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the image and mask pair and apply the transformations

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Image and mask pair
        """
        # Load Image and Mask
        img, mask = self.load_data(idx)

        # Apply Data Augmentations
        img, mask = self.label_handler.apply_transforms(img, mask, self.transforms)

        return img, mask

    def __len__(self) -> int:
        """
        Return length of the dataset

        Returns
        -------
        int
            number of image files
        """
        return len(self.img_files)


if __name__ == "__main__":

    # Define some Transformations
    transforms = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),
        ]
    )

    # Create A Dataset
    Cityscape_DS = BaseDataset(
        root="../../../Datasets/cityscapes",
        img_folder="leftImg8bit_trainvaltest/leftImg8bit/train/*/",
        img_folder_val="leftImg8bit_trainvaltest/leftImg8bit/val/*/",
        label_folder="gtFine_trainvaltest/gtFine/train/*/",
        label_folder_val="gtFine_trainvaltest/gtFine/val/*/",
        label_postfix="labelIds_19classes",
        split="train",
        transforms=transforms,
    )
    print(len(Cityscape_DS))
    # Load Some Data
    img, mask = Cityscape_DS[100]
    print(img.shape, mask.shape)
