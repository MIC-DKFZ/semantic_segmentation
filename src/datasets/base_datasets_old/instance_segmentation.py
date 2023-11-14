from typing import Tuple

import torch
import numpy as np

from src.datasets.base_datasets.base import BaseDataset
from src.datasets.base_datasets.cross_validation import CVDataset


class InstanceDataset(BaseDataset):
    def apply_transforms(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms to image and mask

        Parameters
        ----------
        img: np.ndarray
        mask: np.ndarray

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Image and mask pair
        """
        if self.transforms is not None:
            # Need to Catch empty masks
            empty_mask = len(mask) == 0

            mask = mask if empty_mask else mask.transpose((1, 2, 0))
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]  # / 255
            mask = transformed["mask"]

            mask = mask if empty_mask else mask.permute(2, 0, 1)

        return img, mask

    def load_mask(self, idx):
        """
        Load a single mask (self.mask_files[idx]).

        Returns
        -------
        np.ndarray
            Mask if shape [w,h]
        """
        mask = super().load_mask(idx)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        return masks

    def __getitem__(self, idx):
        """
        Get the image and mask pair and apply the transformations

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Image and mask pair
        """
        # load images and masks + apply transformations
        img, masks = self.load_data(idx)
        img, masks = self.apply_transforms(img, masks)

        # Remove masks which are empty after transformations, caused by spatial transformations
        empty = [bool(torch.any(mask)) for mask in masks]
        masks = masks[empty]

        # get bounding box coordinates for each mask
        boxes = []
        for mask in masks:
            pos = np.where(mask)
            xmin = int(np.min(pos[1]))
            xmax = int(np.max(pos[1]))
            ymin = int(np.min(pos[0]))
            ymax = int(np.max(pos[0]))
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Catch Empty Masks
        if len(masks) == 0:
            area = torch.tensor([])
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # Remove masks with area 0, occur e.g. on border when bb is just a line
            empty = area == 0
            masks = masks[~empty]
            boxes = boxes[~empty]
            area = area[~empty]

        # During Training MaskRCNN can not handle empty masks, so we have to catch it here
        if len(masks) == 0 and self.split == "train":
            return self.__getitem__(np.random.randint(0, self.__len__()))

        # For binary case only a list of 1
        labels = torch.ones((len(masks),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["area"] = area

        return img, target


class InstanceCVDataset(InstanceDataset, CVDataset):
    pass
