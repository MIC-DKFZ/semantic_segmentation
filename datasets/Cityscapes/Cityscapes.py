import os
import glob
from collections import namedtuple

import torch
import torchvision.utils

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

from utils.visualization import show_data
from utils.utils import get_logger

log = get_logger(__name__)

# some parts are taken from here:
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

CityscapesClass = namedtuple(
    "CityscapesClass",
    [
        "name",
        "id",
        "train_id",
        "category",
        "category_id",
        "has_instances",
        "ignore_in_eval",
        "color",
    ],
)
# 34 classes
classes_34 = [
    CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
]

# 19 classes
classes_19 = [
    CityscapesClass("road", 0, 0, "flat", 1, False, False, (128, 64, 128)),
    CityscapesClass("sidewalk", 1, 1, "flat", 1, False, False, (244, 35, 232)),
    CityscapesClass("building", 2, 2, "construction", 2, False, False, (70, 70, 70)),
    CityscapesClass("wall", 3, 3, "construction", 2, False, False, (102, 102, 156)),
    CityscapesClass("fence", 4, 4, "construction", 2, False, False, (190, 153, 153)),
    CityscapesClass("pole", 5, 5, "object", 3, False, False, (153, 153, 153)),
    CityscapesClass("traffic light", 6, 6, "object", 3, False, False, (250, 170, 30)),
    CityscapesClass("traffic sign", 7, 7, "object", 3, False, False, (220, 220, 0)),
    CityscapesClass("vegetation", 8, 8, "nature", 4, False, False, (107, 142, 35)),
    CityscapesClass("terrain", 9, 9, "nature", 4, False, False, (152, 251, 152)),
    CityscapesClass("sky", 10, 10, "sky", 5, False, False, (70, 130, 180)),
    CityscapesClass("person", 11, 11, "human", 6, True, False, (220, 20, 60)),
    CityscapesClass("rider", 12, 12, "human", 6, True, False, (255, 0, 0)),
    CityscapesClass("car", 13, 13, "vehicle", 7, True, False, (0, 0, 142)),
    CityscapesClass("truck", 14, 14, "vehicle", 7, True, False, (0, 0, 70)),
    CityscapesClass("bus", 15, 15, "vehicle", 7, True, False, (0, 60, 100)),
    CityscapesClass("train", 16, 16, "vehicle", 7, True, False, (0, 80, 100)),
    CityscapesClass("motorcycle", 17, 17, "vehicle", 7, True, False, (0, 0, 230)),
    CityscapesClass("bicycle", 18, 18, "vehicle", 7, True, False, (119, 11, 32)),
]

# mapping from 34 class setting to 19 classes
ignore_label = 255
label_mapping = {
    -1: ignore_label,
    0: ignore_label,
    1: ignore_label,
    2: ignore_label,
    3: ignore_label,
    4: ignore_label,
    5: ignore_label,
    6: ignore_label,
    7: 0,
    8: 1,
    9: ignore_label,
    10: ignore_label,
    11: 2,
    12: 3,
    13: 4,
    14: ignore_label,
    15: ignore_label,
    16: ignore_label,
    17: 5,
    18: ignore_label,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: ignore_label,
    30: ignore_label,
    31: 16,
    32: 17,
    33: 18,
}


# cityscapes dataset class
class Cityscapes_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transforms=None):
        # providing the possibility to have data and labels at different locations
        if isinstance(root, str):
            root_imgs = root
            root_labels = root
        else:
            root_imgs = root.IMAGES
            root_labels = root.LABELS

        # no test dataset for cityscapes so return the validation set instead
        if split == "test":
            split = "val"

        # building the paths
        imgs_path = os.path.join(
            root_imgs, "leftImg8bit_trainvaltest", "leftImg8bit", split, "*", "*_leftImg8bit.png"
        )
        masks_path = os.path.join(
            root_labels, "gtFine_trainvaltest", "gtFine", split, "*", "*_gt*_labelIds_19classes.png"
        )
        # elif num_classes==34:
        #    masks_path=os.path.join( root_labels,"gtFine_trainvaltest" , "gtFine" , split , "*" , "*_gt*_labelIds.png" )

        # save all paths in lists
        self.imgs = list(sorted(glob.glob(imgs_path)))
        self.masks = list(sorted(glob.glob(masks_path)))

        self.transforms = transforms
        log.info(
            "Dataset: Cityscape %s - %s images - %s masks", split, len(self.imgs), len(self.masks)
        )

    def __getitem__(self, idx):
        # read image (opencv read images in bgr) and mask
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[idx], -1)

        # apply albumentations transforms
        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


def viz_color_encoding():
    # litte helper function to visualize the color-class encoding
    width = 700
    height = 60

    def sub_viz(classes):
        num = len(classes)
        img = np.zeros((num * height, width, 3), np.uint8)

        for index, c in enumerate(classes):
            img[index * height : (height + 1) * height, :] = c.color
            cv2.putText(
                img,
                str(index) + ".",
                (10, (index) * height + int(height * 0.75)),
                cv2.FONT_HERSHEY_COMPLEX,
                1.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                c.name,
                (150, (index) * height + int(height * 0.75)),
                cv2.FONT_HERSHEY_COMPLEX,
                1.5,
                (255, 255, 255),
                2,
            )
            print(c.name, c.color)
        for index in range(1, num):
            cv2.line(img, (0, index * height), (width, index * height), (255, 255, 255))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    img_full = sub_viz(classes_34)
    classes_eval = [c for c in classes_34 if c.ignore_in_eval == False]
    img_eval = sub_viz(classes_eval)

    cv2.imwrite("Cityscape_color_encoding_full.png", img_full)
    cv2.imwrite("Cityscape_color_encoding.png", img_eval)


if __name__ == "__main__":

    # define some transforms
    transforms = A.Compose(
        [
            # A.RandomCrop(width=768, height=768),
            A.RandomScale(scale_limit=(-0.5, 1), always_apply=True, p=1.0),
            # A.PadIfNeeded(min_height=768,min_width=768),
            # A.Resize(p=1.0,width=1024, height=512),
            A.RandomCrop(width=1024, height=512, always_apply=True, p=1.0),
            # A.ColorJitter(brightness=9,contrast=0,saturation=0,hue=0),
            A.RGBShift(p=1, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            A.GaussianBlur(),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),
        ]
    )
    print(transforms)
    # print(transforms)
    # A.save(transforms,"config/transform_test.yaml",data_format='yaml')

    # load a dataset
    cityscapesPath = "/home/l727r/Desktop/Datasets/cityscapes"
    Cityscape_train = Cityscapes_dataset(cityscapesPath, "train", transforms=transforms)

    # load some data and visualie it
    img, mask = Cityscape_train[100]
    print(img.shape)
    print(torch.unique(mask))

    color_mapping = [x.color for x in classes_19]
    out = show_data(
        img=img,
        mask=mask,
        alpha=0.0,
        black=[255],
        color_mapping=[x.color for x in classes_19],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    out.show()
    # out.save("out.png")
