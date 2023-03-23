import torch
import os
import pandas as pd
import cv2
import numpy as np


class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root = root
        self.df = pd.read_csv(open(os.path.join(root, "annotations.csv")))
        files = self.df.file.unique()
        files.sort()
        train_files = files[:23]
        val_files = files[23:]

        if split == "train":
            self.files = train_files
        elif split == "val" or split == "test":
            self.files = val_files
        self.split = split
        self.transforms = transforms

        # min_vals = []
        # max_vals = []
        # for file in train_files:
        #     img = cv2.imread(os.path.join(self.root, "ground_truth", "images", file), -1)
        #     min_vals.append(img.min())
        #     max_vals.append(img.max())
        # self.min_val = min(min_vals)
        # self.max_val = max(max_vals)
        # print(self.min_val, self.max_val)
        print("Solar_Hydrogen Dataset: {} Files found for {} set".format(len(self.files), split))

        # print("X: {}".format(self.files))

    def __getitem__(self, idx):
        # Select the file and the corresponding entry in the df
        file = self.files[idx]
        rows = self.df[self.df["file"] == file]

        # Open the image, scale it to RGB range and make it a 3 channel image (for data augmentations + pretrained model)
        img = cv2.imread(os.path.join(self.root, "ground_truth", "images", file), -1)
        # print(img.min(), img.max())
        img = (img - img.min()) / (img.max() - img.min()) * 255
        # img = (img - self.min_val) / (self.max_val - self.min_val) * 255
        img = np.dstack((img, img, img))
        img = img.astype(np.uint8)

        # Create the masks by drawing a circle with the given center point and radius
        masks = []
        for x, y, r in zip(rows.x, rows.y, rows.radius):
            mask = np.zeros((img.shape[0], img.shape[1]))
            cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
            masks.append(mask)
        masks = np.array(masks, dtype=np.uint8)

        if self.transforms is not None:
            masks = masks.transpose((1, 2, 0))
            transformed = self.transforms(image=img, mask=masks)
            img = transformed["image"]  # / 255
            masks = transformed["mask"].permute(2, 0, 1)

        # Remove masks which are empty after transformations
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

        if len(masks) == 0:
            areas = torch.tensor([])
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Compute the area of each bounding box
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # Remove masks with area 0, occur e.g. on border when bb is just a line
            empty = areas == 0
            masks = masks[~empty]
            boxes = boxes[~empty]
            areas = areas[~empty]

        # During Training MaskRCNN has problems with Empty Images (which contains not masks)
        # Even if no empty images in dataset, this can happen because of spatial data augmentations
        if len(masks) == 0 and self.split == "train":
            return self.__getitem__(np.random.randint(0, self.__len__()))

        # Fill in the rest of needed information and put it together into target dict
        num_objs = len(masks)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        # target["image_id"] = image_id
        target["area"] = areas
        # target["iscrowd"] = iscrowd

        return img, target

    def __getitem___s(self, idx):
        # reading images and masks as numpy arrays
        file = self.files[idx]
        rows = self.df[self.df["file"] == file]

        # img_org = cv2.imread(os.path.join(self.root, "ground_truth", "images", file))
        img_org = cv2.imread(os.path.join(self.root, "ground_truth", "images", file), -1)
        img_org = (img_org - img_org.min()) / (img_org.max() - img_org.min()) * 255
        img_org = np.dstack((img_org, img_org, img_org))

        masks = []
        for x, y, r in zip(rows.x, rows.y, rows.radius):
            mask = np.zeros((img_org.shape[0], img_org.shape[1]))
            cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
            masks.append(mask)
        masks = np.array(masks, dtype=np.uint8)
        # img_org = img_org.astype(np.float32)
        img_org = img_org.astype(np.uint8)
        masks1 = masks.transpose((1, 2, 0))
        while True:
            if self.transforms is not None:
                transformed = self.transforms(image=img_org, mask=masks1)
                img = transformed["image"] / 255
                masks = transformed["mask"].permute(2, 0, 1)

            # Remove empty masks
            x = [bool(torch.any(mask)) for mask in masks]
            masks = masks[x]

            num_objs = len(masks)
            if num_objs <= 0:
                continue

            boxes = []
            masks_np = np.array(masks)
            for i in range(num_objs):
                pos = np.where(masks_np[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # print(boxes)
            labels = torch.ones((num_objs,), dtype=torch.int64)

            image_id = torch.tensor([idx])

            if len(boxes) == 0:
                area = torch.tensor([])
            else:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if 0 in area:
                continue
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            break

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.files)
