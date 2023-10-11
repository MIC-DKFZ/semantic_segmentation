from os.path import join
import os
from typing import List, Tuple, Optional
import json

import numpy as np

from src.datasets.base_datasets.base import BaseDataset
from src.utils.utils import get_logger
from src.utils.dataset_utils import random_scale_crop, keypoint_scale_crop

log = get_logger(__name__)


class SamplingDataset(BaseDataset):
    def __init__(
        self,
        num_classes: int,
        patch_size: Tuple[int, int] = (512, 512),
        scale_limit: Tuple[int, int] = (0, 0),
        num_sampling_points: int = 10000,
        batch_size: int = 1,
        steps_per_epoch: int = 250,
        random_sampling: float = 0.3,
        class_probabilities: Optional[List[float]] = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        num_classes: int
            Number of Classes in the Dataset
        patch_size: Tuple[int, int], optional
            patch or crop size which should be used for sampling
        scale_limit: Tuple[int, int], optional
            [min_scale,max_scale] in albumentations definition (0 means no scaling)
        num_sampling_points: int, optional
            number of occurrences which should be sampled for each class
        batch_size: int, optional
            batch size -> needed for finding number of steps
        steps_per_epoch: int, optional
            how many steps should be done per epoch
        random_sampling: float, optional
            probability for random sampling
        class_probabilities: Optional[List[float]], optional
            probabilities for sampling each class, if None each class has the same probability (balanced sampling)
        kwargs
        """
        # Sampling related parameters
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.scale_limit = scale_limit
        self.random_sampling = random_sampling
        self.class_probabilities = class_probabilities
        self.steps_per_epoch = batch_size * steps_per_epoch
        self.num_sampling_points = num_sampling_points

        # Normalize the class_probabilities if given to sum up to 1
        if self.class_probabilities:
            self.class_probabilities = self.class_probabilities / sum(self.class_probabilities)

        super().__init__(num_classes=num_classes, **kwargs)

    def setup(self):
        """
        Get Paths to the Image and Mask Files
        Preprocessing:
            Sample occurrences for each class
        Load class_occurrences.json (only for training):
            Contains {self.num_sampling_points} points for each class
            Filer out all images which are not in train split
        """
        super().setup()
        # Run Preprocessing
        self.preprocessing_sampling()
        # Preprocess the information about which classes occur in which image
        # we only need the ones of the current split
        if self.split == "train":
            with open(join(self.root, "class_occurrences.json"), "r") as file:
                self.class_occurrences = json.load(file)
            mask_names = [
                os.path.split(mask)[-1].replace(self.dtype_mask, "") for mask in self.mask_files
            ]
            self.class_occurrences = [
                [item for item in O_set if item in mask_names] for O_set in self.class_occurrences
            ]

    def preprocessing_sampling(self) -> None:
        """
        Preprocessing (only if class_occurrences.json not already exists):
            Sample {self.num_sampling_points} points for each class and save for each file
            For each class save all images which contain the class and save
        """
        # Only run preprocessing_sampling if class_occurrences.json not already exists
        if os.path.exists(join(self.root, "class_occurrences.json")):
            return

        log.info(f"Dataset Start with Preprocessing for Sampling")

        os.makedirs(join(self.root, "class_locations"), exist_ok=True)

        class_occurrences = [[] for _ in range(self.num_classes)]
        for idx, mask_file in enumerate(self.mask_files):
            sampled_points = {}
            mask_name = os.path.split(mask_file)[-1].replace(self.dtype_mask, "")

            # Load mask, get present Classes and catch ignore label
            mask = self.load_mask(idx)
            unique_vals = np.unique(mask)
            unique_vals = [val for val in unique_vals if self.num_classes > val >= 0]

            for unique_val in unique_vals:
                # Get pixels with class ID, and select a defined number randomly
                x, y = np.where(mask == unique_val)
                index = np.random.choice(
                    np.arange(len(x)),
                    min(len(x), self.num_sampling_points),
                    replace=False,
                )
                x = x[index]
                y = y[index]

                # Save them to the dict
                sampled_points[str(unique_val)] = {"x": x.tolist(), "y": y.tolist()}

            # Save sampled points to json file with format: {Class-ID: {'x': x, 'y': y}}
            with open(join(self.root, "class_locations", mask_name + ".json"), "w") as file:
                json.dump(sampled_points, file)

            # Save information about which classes occur in which file
            for c in unique_vals:
                class_occurrences[c].append(mask_name)

        # Summary of which classes occur in which files in format {Class-ID: [files]}
        with open(join(self.root, "class_occurrences.json"), "w") as file:
            json.dump(class_occurrences, file)

    def load_data_random(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly sample an image
        Randomly sample a patch from the image

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Image and mask pair
        """
        idx = np.random.randint(0, len(self.img_files))
        img, mask = super().load_data(idx)
        if self.patch_size is not None:
            img, mask = random_scale_crop(img, mask, self.patch_size, self.scale_limit)
        return img, mask

    def load_data_sampled(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly sample a class
        Randomly sample an image which contains the class
        Randomly sample a point in the image which belongs to the class
        Crop around the selected point

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Image and mask pair
        """
        # 1. Randomly select a Class
        class_id = np.random.choice(
            np.arange(0, len(self.class_occurrences)), p=self.class_probabilities
        )

        # 2. Randomly select an Image containing this Class (from preprocessing)
        img_file = np.random.choice(self.class_occurrences[class_id])

        # 3. Randomly select a Point in the Image which belongs to the Class (from preprocessing)
        with open(join(self.root, "class_locations", img_file + ".json"), "r") as file:
            data = json.load(file)[str(class_id)]
            pt_idx = np.random.randint(0, len(data["x"]))
            x, y = data["x"][pt_idx], data["y"][pt_idx]

        # 4. Find the index of the selected file in self.mask_files and Load Image and Mask
        idx = [i for i, s in enumerate(self.mask_files) if img_file in s][0]
        img, mask = super().load_data(idx)

        # 5. Center Crop the image by the selected Point
        if self.patch_size is not None:
            img, mask = keypoint_scale_crop(img, mask, self.patch_size, (x, y), self.scale_limit)
        return img, mask

    def load_data(self, idx: int) -> tuple:
        """
        Load an image and mask pair.
        During test and val: base data loading
        Randomly decide (by {self.random_sampling}) if random and sampled data loading

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Image and mask pair
        """
        if self.split != "train":
            return super().load_data(idx)
        elif np.random.random() <= self.random_sampling:
            return self.load_data_random()
        else:
            return self.load_data_sampled()

    def __len__(self) -> int:
        """
        Return length of the data/steps per epoch

        Returns
        -------
        int
            for val and test the number of files
            for train the number of steps per epoch
        """
        return self.steps_per_epoch if self.split == "train" else len(self.img_files)
