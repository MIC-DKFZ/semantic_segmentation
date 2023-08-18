import cv2
import torch
import json
import os
from src.utils.utils import get_logger

log = get_logger(__name__)


class Computing_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms, fold=0):
        # get your data for the corresponding split
        if split == "test":
            split = "val"

        with open(os.path.join(root, "splits_final.json")) as f:
            splits = json.load(f)[fold][split]

        self.imgs = [os.path.join(root, "imagesTr", name + "_0000.png") for name in splits]
        self.masks = [os.path.join(root, "labelsTr", name + ".png") for name in splits]

        # save the transformations
        self.transforms = transforms

    def __getitem__(self, idx):
        # reading images and masks as numpy arrays
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order
        mask = cv2.imread(self.masks[idx], -1)

        # thats how you apply Albumentations transformations
        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)
