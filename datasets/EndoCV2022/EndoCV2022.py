import os

import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from utils.visualization import show_data
from utils.utils import get_logger

log = get_logger(__name__)

ignore_label = 255

# Data shape [3, 1064, 1440],[3, 1072, 1728],[3, 1024, 1280],[3, 1080, 1920],[3, 720, 1280],[3, 1072, 1704]

PALETTE = [[0, 0, 0], [255, 0, 0]]


class EndoCV2022_dataset(torch.utils.data.Dataset):
    def __init__(self, root, fold, split="train", transforms=None):
        if split == "test":
            split = "val"

        data = pd.read_csv(os.path.join(root, "polypgen_cleaned_4CV.csv"))

        if fold == "all":
            data = data
        elif split == "val":
            data = data[data.fold == fold]
        elif split == "train":
            data = data[data.fold != fold]

        self.imgs = []
        self.masks = []

        folder = "clean_PolypGen2.0"

        for i, d in data.iterrows():
            self.imgs.append(os.path.join(root, folder, d.vid_folder, "images", d.image_id))
            self.masks.append(os.path.join(root, folder, d.vid_folder, "masks", d.Mask_id))
        self.transforms = transforms
        log.info(
            "Dataset: EncoCV2022 %s - Fold %s - %s images - %s masks",
            split,
            fold,
            len(self.imgs),
            len(self.masks),
        )

    def __getitem__(self, idx):

        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = (cv2.imread(self.masks[idx]) > 20).astype(np.uint8)[:, :, 0]

        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":

    transforms = A.Compose(
        [
            A.RandomScale(scale_limit=(-0.5, 0), always_apply=True, p=1.0),
            A.RGBShift(p=1, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            A.PadIfNeeded(min_height=512, min_width=1024),
            A.RandomCrop(height=512, width=1024),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    root_path = "/home/l727r/Desktop/endocv2022/official EndoCV2022 dataset"
    EndoCV = EndoCV2022_dataset(root=root_path, fold=1, split="train", transforms=transforms)
    print(len(EndoCV))
    EndoCV = EndoCV2022_dataset(root=root_path, fold=1, split="val", transforms=transforms)
    print(len(EndoCV))

    img, mask = EndoCV[150]
    out = show_data(
        img,
        mask,
        alpha=0.5,
        color_mapping=[[0, 0, 0], [255, 0, 0]],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    out.show()
