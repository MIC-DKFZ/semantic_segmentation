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
    def get_mask_files(self, split) -> list:

        mask_files = super().get_mask_files(split)
        mask_files = [mask.rsplit("_", 1)[0] for mask in mask_files]
        mask_files = np.unique(mask_files)

        return list(sorted(mask_files))


class MultilabelSamplingDataset(MultilabelDataset, SamplingDataset):
    pass


class MultilabelCVDataset(MultilabelDataset, CVDataset):
    pass


class MultilabelSamplingCVDataset(MultilabelSamplingDataset, CVDataset):
    pass
