from os.path import join, split
import os
import json

import numpy as np

from datasets.Base_Datasets.base import Base_Dataset
from datasets.Base_Datasets.cross_validation import CV_Dataset
from datasets.Base_Datasets.sampling import Sampling_Dataset
from src.utils import get_logger
from src.dataset_utils import random_scale_crop, keypoint_scale_crop
import cv2

cv2.setNumThreads(0)


log = get_logger(__name__)


class Multilabel_Dataset(Base_Dataset):
    def apply_transforms(self, img: np.ndarray, mask: np.ndarray) -> tuple:
        if self.transforms is not None:
            mask = mask.transpose((1, 2, 0))
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1)  # .long()
        return img, mask


class Multilabel_Sampling_Dataset(Multilabel_Dataset, Sampling_Dataset):
    def preprocessing_sampling(self) -> None:
        # Only run preprocessing_sampling if class_occurrences.json not already exists
        if os.path.exists(join(self.root, "class_occurrences.json")):
            return

        log.info(f"Dataset Start with Preprocessing for Sampling")

        os.makedirs(join(self.root, "class_locations"), exist_ok=True)

        class_occurrences = [[] for _ in range(self.num_classes)]
        for idx, mask_file in enumerate(self.mask_files):
            sampled_points = {}
            mask_name = split(mask_file)[-1].replace(self.dtype, "")

            # Load mask, get present Classes and catch ignore label
            mask = self.load_mask(idx)
            not_empty_classes = [i for i, m in enumerate(mask) if np.any(m)]
            for class_id in not_empty_classes:
                # Get pixels with class ID, and select a defined number randomly
                x, y = np.where(mask[class_id] == 1)
                index = np.random.choice(
                    np.arange(len(x)),
                    min(len(x), self.num_sampling_points),
                    replace=False,
                )
                x = x[index]
                y = y[index]

                # Save them to the dict
                sampled_points[class_id] = {"x": x.tolist(), "y": y.tolist()}

            # Save sampled points to json file with format: {Class-ID: {'x': x, 'y': y}}
            with open(join(self.root, "class_locations", mask_name + ".json"), "w") as file:
                json.dump(sampled_points, file)

            # Save information about which classes occur in which file
            for c in not_empty_classes:
                class_occurrences[c].append(mask_name)

        # Summary of which classes occur in which files in format {Class-ID: [files]}
        with open(join(self.root, "class_occurrences.json"), "w") as file:
            json.dump(class_occurrences, file)

    def load_data_random(self):
        idx = np.random.randint(0, len(self.img_files))
        img, mask = Multilabel_Dataset.load_data(self, idx)
        mask = mask.transpose((1, 2, 0))
        img, mask = random_scale_crop(img, mask, self.patch_size, self.scale_limit)
        mask = mask.transpose((2, 0, 1))
        return img, mask

    def load_data_sampled(self):
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
        img, mask = Multilabel_Dataset.load_data(self, idx)

        # 5. Center Crop the image by the selected Point
        mask = mask.transpose((1, 2, 0))
        img, mask = keypoint_scale_crop(img, mask, self.patch_size, (x, y), self.scale_limit)
        mask = mask.transpose((2, 0, 1))

        return img, mask


class Multilabel_CV_Dataset(Multilabel_Dataset, CV_Dataset):
    pass


class Multilabel_Sampling_CV_Dataset(Multilabel_Sampling_Dataset, CV_Dataset):
    pass
