from os.path import join, split
import os
import json
from typing import Tuple
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
from src.datasets.base_datasets.base import BaseDataset
from src.datasets.base_datasets.cross_validation import CVDataset
from src.datasets.base_datasets.sampling import SamplingDataset
from src.utils.utils import get_logger
from src.utils.dataset_utils import random_scale_crop, keypoint_scale_crop
import cv2

cv2.setNumThreads(0)


log = get_logger(__name__)


class MultilabelDataset(BaseDataset):
    def __init__(self, num_classes, **kwargs):

        self.num_classes = num_classes
        super().__init__(num_classes=num_classes, **kwargs)

    # def get_img_files(self) -> list:
    #     img_files = [
    #         "/media/l727r/data/Datasets/dacl10k/dacl10k_dataset/images_train/dacl10k_v2_train_0001.jpg"
    #     ]
    #     if self.split == "train":
    #         return img_files * 50
    #     return img_files
    #
    #
    #     return list(sorted(mask_files))
    #
    # def get_mask_files(self) -> list:
    #     mask_files = [
    #         "/media/l727r/data/Datasets/dacl10k/dacl10k_dataset/labels_train/dacl10k_v2_train_0001"
    #     ]
    #     if self.split == "train":
    #         return mask_files * 50
    #     return mask_files

    def get_mask_files(self) -> list:

        mask_files = super().get_mask_files()
        mask_files = [mask.rsplit("_", 1)[0] for mask in mask_files]
        mask_files = np.unique(mask_files)

        return list(sorted(mask_files))

    def load_mask(self, idx):
        masks = []
        mask_file = self.mask_files[idx]
        for i in range(0, self.num_classes):
            mask = cv2.imread(f"{mask_file}{self.label_postfix}_{i}.png", -1)
            masks.append(mask)
        return np.array(masks, dtype=np.uint8)

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
            mask = mask.transpose((1, 2, 0))
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1)  # .long()
        return img, mask


class MultilabelSamplingDataset(MultilabelDataset, SamplingDataset):
    def preprocess_sample(self, mask_file, idx):
        sampled_points = {}
        mask_name = split(mask_file)[-1].replace(self.dtype, "")

        # Load mask, get present Classes and catch ignore label
        mask = self.load_mask(idx)
        not_empty_classes = [i for i, m in enumerate(mask) if np.any(m)]
        # print(not_empty_classes)
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

        return mask_name, not_empty_classes

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
        pool = multiprocessing.Pool(processes=8)
        results_list = []
        for idx, mask_file in enumerate(self.mask_files):

            results_list.append(
                pool.starmap_async(
                    self.preprocess_sample,
                    (
                        (
                            mask_file,
                            idx,
                        ),
                    ),
                )
            )

        results_list = [a.get() for a in tqdm(results_list, desc="Preprocessing Sampling...")]
        pool.close()
        pool.join()
        class_occurrences = [[] for _ in range(self.num_classes)]
        for result in results_list:
            name, not_empty = result[0]
            for r in not_empty:
                class_occurrences[r].append(name)
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
        img, mask = MultilabelDataset.load_data(self, idx)

        if self.patch_size is not None:
            mask = mask.transpose((1, 2, 0))
            img, mask = random_scale_crop(img, mask, self.patch_size, self.scale_limit)
            mask = mask.transpose((2, 0, 1))
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
        img, mask = MultilabelDataset.load_data(self, idx)

        # 5. Center Crop the image by the selected Point
        if self.patch_size is not None:
            mask = mask.transpose((1, 2, 0))
            img, mask = keypoint_scale_crop(img, mask, self.patch_size, (x, y), self.scale_limit)
            mask = mask.transpose((2, 0, 1))
        return img, mask


class MultilabelCVDataset(MultilabelDataset, CVDataset):
    pass


class MultilabelSamplingCVDataset(MultilabelSamplingDataset, CVDataset):
    pass
