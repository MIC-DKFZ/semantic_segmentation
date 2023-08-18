import os
import pandas as pd
import cv2
import numpy as np

from src.utils.utils import get_logger
from datasets.Base_Datasets.instance_segmentation import Instance_Dataset_CV


log = get_logger(__name__)


class Solar_Hydrogen_Dataset(Instance_Dataset_CV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(self.root)
        self.csv_file = "annotations.csv"
        self.df = pd.read_csv(open(os.path.join(self.root, self.csv_file)))

    def get_mask_files(self) -> list:
        return self.img_files

    def load_data(self, idx):
        # Load Image
        img = cv2.imread(self.img_files[idx], -1)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = np.dstack((img, img, img)).astype(np.uint8)

        # Load Mask Data
        file_name = os.path.split(self.img_files[idx])[-1]
        rows = self.df[self.df["file"] == file_name]

        # Create Masks
        masks = []
        for x, y, r in zip(rows.x, rows.y, rows.radius):
            mask = np.zeros((img.shape[0], img.shape[1]))
            cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
            masks.append(mask)
        masks = np.array(masks, dtype=np.uint8)

        return img, masks
