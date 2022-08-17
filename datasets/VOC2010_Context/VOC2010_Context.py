import os
import glob

import torch

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from src.visualization_utils import show_data
from src.utils import get_logger

log = get_logger(__name__)

### NAME OF ALL CLASSES ###
CLASSES = (
    "background",
    "aeroplane",
    "bag",
    "bed",
    "bedclothes",
    "bench",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bottle",
    "building",
    "bus",
    "cabinet",
    "car",
    "cat",
    "ceiling",
    "chair",
    "cloth",
    "computer",
    "cow",
    "cup",
    "curtain",
    "dog",
    "door",
    "fence",
    "floor",
    "flower",
    "food",
    "grass",
    "ground",
    "horse",
    "keyboard",
    "light",
    "motorbike",
    "mountain",
    "mouse",
    "person",
    "plate",
    "platform",
    "pottedplant",
    "road",
    "rock",
    "sheep",
    "shelves",
    "sidewalk",
    "sign",
    "sky",
    "snow",
    "sofa",
    "table",
    "track",
    "train",
    "tree",
    "truck",
    "tvmonitor",
    "wall",
    "water",
    "window",
    "wood",
)

### COLORMAPPING FOR EACH CLASS ###
PALETTE = [
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
]


class VOC2010_Context_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", num_classes=60, ignore_index=255, transforms=None):
        # providing the possibility to have data and labels at different locations
        if isinstance(root, str):
            root_imgs = root
            root_labels = root
        else:
            root_imgs = root.IMAGES
            root_labels = root.LABELS

        self.split = split
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        if split == "test":
            split = "val"
        imgs_path = os.path.join(root_imgs, "Images", split, "*.jpg")

        masks_path = os.path.join(root_labels, "Annotations", split, "*.png")

        self.imgs = list(sorted(glob.glob(imgs_path)))
        self.masks = list(sorted(glob.glob(masks_path)))

        self.transforms = transforms
        # log.info("Dataset: VOC2010_Context %s - %s images - %s masks",split,  len(self.imgs),len(self.masks))
        print("Dataset: VOC2010_Context", split, len(self.imgs), len(self.masks))

    def reduce_num_classes(self, mask):
        # exclude background class
        mask = mask - 1
        mask[mask == -1] = self.ignore_index
        return mask

    def __getitem__(self, idx):
        # read image (opencv read images in bgr) and mask
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[idx], -1)

        # reduce the number of classes is specified in the config - yes by default
        if self.num_classes == 59:
            mask = self.reduce_num_classes(mask)

        # apply albumentations transforms
        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":

    transforms = A.Compose(
        [
            # A.RandomCrop(width=768, height=768),
            A.SmallestMaxSize(max_size=520),
            # A.RandomScale(scale_limit=(-0.5, 1), always_apply=True, p=1.0),
            # A.RGBShift(p=1,r_shift_limit=10,g_shift_limit=10,b_shift_limit=10),
            A.RandomScale(scale_limit=(-0.5, 1), always_apply=True, p=1.0),
            A.PadIfNeeded(min_height=520, min_width=520, border_mode=0, value=0, mask_value=255),
            # A.Resize(p=1.0,width=480, height=480),
            # A.RandomCrop(width=520, height=520,always_apply=True,p=1.0),
            # A.GaussianBlur(p=1),
            # A.ColorJitter(brightness=9,contrast=0,saturation=0,hue=0),
            # A.RGBShift(p=1,r_shift_limit=10,g_shift_limit=10,b_shift_limit=10),
            A.GaussianBlur(p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),
        ]
    )
    print(transforms)
    Path = "/home/l727r/Desktop/Datasets/VOC2010_Context"
    VOC2010_train = VOC2010_Context_dataset(Path, "train", transforms=transforms)

    img, mask = VOC2010_train[465]
    print(np.unique(img))
    print(img[:, 100, 100])
    print(img[:, 200, 200])
    print(img[:, 300, 300])

    out = show_data(
        img=img,
        mask=mask,
        alpha=0.5,
        black=[255],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    out.show()
