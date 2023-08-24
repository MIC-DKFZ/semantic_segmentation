import torch
import os
import pandas as pd
import cv2
import numpy as np
import random

from src.utils import get_logger

log = get_logger(__name__)


class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train", transforms=None):
        """
        Initialization of the Dataset Class

        Parameters
        ----------
        root: str
            Root Directory which contains the images and the annotation csv
        split: str (optional)
            which split to use, "train" (for training, used by default)  or "val" (for validation)
        transforms
            transformation which will be applied to the data, you don't have to care about this. If
            you want to change the augmentation pipeline look at config/Readme.md at the
            augmentations part, still nothing to do in this file
        """
        self.root = root
        self.split = split
        self.transforms = transforms
        if split == "test":
            # Name of the subfolder which contains the images, change this if needed
            self.img_folder = "images_test"
            # Name of the csv file which contains the bubble annotations, change if needed
            self.csv_file = "annotations_test.csv"
        elif split == "train" or split == "val":
            # Name of the subfolder which contains the images, change this if needed
            self.img_folder = "images"
            # Name of the csv file which contains the bubble annotations, change if needed
            self.csv_file = "annotations.csv"

        # Read the csv file which contains the annotations
        self.df = pd.read_csv(open(os.path.join(root, self.csv_file)))

        # Get the name of files which are in the csv without duplicates
        files = self.df.file.unique()
        files.sort()
        if split == "test":
            self.files = files
        if split == "train" or split == "val":
            # in our case the images where homogeneous (same crop of the image) and from different
            # source images. If this is not the case for new data it may make sense to shuffle the data
            # for a random distribution of certain characteristics over the train and val set
            # seed=123 # seed is needed since the data should always be shuffled in the same way
            # random.Random(seed).shuffle(files)

            # define the number of validation cases, ~25% of the dataset
            num_val_cases = len(files) // 4

            # Select the files which are used for training and validation
            # Here  just the last num_val_cases are chosen for validation and the rest for training
            train_files = files[:-num_val_cases]
            val_files = files[-num_val_cases:]
            if split == "train":
                self.files = train_files
            elif split == "val":
                self.files = val_files

        log.info("Solar_Hydrogen Dataset: {} Files found for {} set".format(len(self.files), split))

    def __getitem__(self, idx: int):
        """
        This function is called by the dataloader and returns the images as well as the label information
        I think you should not the need anything here if you adopt the script to new data, everything
        which is relevant for you is inside the __init__ method


        Parameters
        ----------
        idx: int
            index of the file to return
        Returns
        -------
        img: torch.Tensor
            Tensor of the image
        target: dict
            Target dict which contains e.g. the masks, labels and bounding boxes
        """
        # Select the file and the corresponding entry in the df
        file = self.files[idx]
        rows = self.df[self.df["file"] == file]

        # Open the image, scale it to RGB range and make it a 3 channel image
        # Needed for data augmentations and the use of pretrained model
        img = cv2.imread(os.path.join(self.root, self.img_folder, file), -1)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = np.dstack((img, img, img))
        img = img.astype(np.uint8)

        # Create the masks by drawing a circle with the given center point and radius for each mask
        # and stacking them together
        masks = []
        for x, y, r in zip(rows.x, rows.y, rows.radius):
            mask = np.zeros((img.shape[0], img.shape[1]))
            cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
            masks.append(mask)
        masks = np.array(masks, dtype=np.uint8)

        # Apply the transformations, mask transpose back and forth is necessary to make
        # albumentations transformations work with instance segmentation masks
        if self.transforms is not None:
            masks = masks.transpose((1, 2, 0))
            transformed = self.transforms(image=img, mask=masks)
            img = transformed["image"]  # / 255
            masks = transformed["mask"].permute(2, 0, 1)

        # Remove masks which are empty after transformations, caused by spatial transformations
        empty = [bool(torch.any(mask)) for mask in masks]
        masks = masks[empty]

        # Compute the Bounding Boxes for each mask
        boxes = []
        for mask in masks:
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Compute the area of each bounding box, needed for metric computation
        if len(masks) == 0:
            areas = torch.tensor([])
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # Remove masks with area 0, occur e.g. on border when bb is just a line
            empty = areas == 0
            masks = masks[~empty]
            boxes = boxes[~empty]
            areas = areas[~empty]

        # During Training MaskRCNN has problems with Empty Images (which contains not masks)
        # Even if no empty images in dataset, this can happen because of spatial data augmentations
        # Should not occur often, but when it happens we just select another image
        if len(masks) == 0 and self.split == "train":
            return self.__getitem__(np.random.randint(0, self.__len__()))

        # Define the label of each mask, since we only have one class (bubble) it's just a list of 1
        labels = torch.ones((len(masks),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["area"] = areas

        return img, target

    def __len__(self):
        """
        Method which returns the number of files inside the dataset, is needed for the dataloader
        """
        return len(self.files)
