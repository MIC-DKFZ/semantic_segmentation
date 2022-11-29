import torch
import os
import pandas as pd
import cv2
import numpy as np


class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root = root
        self.df = pd.read_csv(open(os.path.join(root, "annotations.csv")))
        self.files = self.df.file.unique()
        self.files.sort()
        if split == "train":
            self.files = self.files[:23]
        elif split == "val" or split == "test":
            self.files = self.files[23:]
        self.transforms = transforms
        print("Solar_Hydrogen Dataset: {} Files found for {} set".format(len(self.files), split))

        # print("X: {}".format(self.files))

    def __getitem__(self, idx):
        # reading images and masks as numpy arrays
        file = self.files[idx]
        rows = self.df[self.df["file"] == file]

        # img = cv2.imread(os.path.join(self.root, "Data", "images", file + ".tif"), -1)
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
