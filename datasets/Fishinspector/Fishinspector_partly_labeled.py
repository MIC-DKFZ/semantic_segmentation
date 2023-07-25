import torch
import json
import os
from src.utils import get_logger
from os.path import join
import numpy as np

from datasets.Fishinspector.Fishinspector import Fishinspector_Base
from datasets.Base_Datasets.multilabel import Multilabel_CV_Dataset, Multilabel_Sampling_CV_Dataset
from src.dataset_utils import random_scale_crop, keypoint_scale_crop

log = get_logger(__name__)


class Fishinspector_PL_Base(Fishinspector_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dorsal_classes = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
        self.lateral_classes = np.array(
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool
        )
        self.trusted_classes = np.array(
            [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=bool
        )
        #   [0, 1, 0?, 0, 1?, 0, 0, 0, 1, 1, 0, 0?, 1, 1, 0, 1] # old version

    def load_mask(self, idx):
        """
        map: all classes are true which are:
            - marked as labeled in the annotated_labels.json & the class has no trust issue
            - class is in the other set
        mask:
            Empty all Masks of the other set which are not empty
        """
        # Load the mask and get file name
        masks = super().load_mask(idx)
        mask_file = self.mask_files[idx]
        # print(mask_file)
        # 1) Get all classes which are marked as labeled in the annotated_labels.sjon
        with open(mask_file.replace("_0000.png", ".json")) as file:
            labeled_classes = json.load(file)["annotated_labels"]
            # print(labeled_classes)
            map_labeled_classes = np.zeros(self.num_classes, dtype=bool)
            map_labeled_classes[labeled_classes] = True

        # 2) Get all empty masks
        map_empty_classes = ~np.any(masks == 1, axis=(1, 2)).astype(bool)

        # 3) Get all classes which are not trusted and marked as labeled but have an empty mask
        map_trust_issue = ~self.trusted_classes & map_labeled_classes & map_empty_classes

        # 4) Get all classes of the other set
        if "Dorsal" in os.path.split(mask_file)[-1]:
            map_other_set = self.lateral_classes.copy()
        elif "Lateral" in os.path.split(mask_file)[-1]:
            map_other_set = self.dorsal_classes.copy()

        # 5) Combine: marked as labeled or in the other set but have no trust issue
        map = (map_labeled_classes | map_other_set) & ~map_trust_issue
        map = torch.tensor(map)
        # 6) Empty all Masks of the other set which are not empty
        masks[~map_empty_classes & map_other_set] = np.zeros(masks[0].shape)

        return masks, map

    def load_data(self, idx: int) -> tuple:
        img = self.load_img(idx)
        mask, map = self.load_mask(idx)

        return img, mask, map

    def __getitem__(self, idx):

        img, mask, map = self.load_data(idx)
        img, mask = self.apply_transforms(img, mask)

        return img, mask, map


class Fishinspector_dataset_PL(Multilabel_CV_Dataset, Fishinspector_PL_Base):
    pass


class Fishinspector_sampling_dataset_PL(Multilabel_Sampling_CV_Dataset, Fishinspector_PL_Base):
    def load_data_random(self):
        idx = np.random.randint(0, len(self.img_files))
        img, mask, map = Fishinspector_dataset_PL.load_data(self, idx)
        mask = mask.transpose((1, 2, 0))
        img, mask = random_scale_crop(img, mask, self.patch_size, self.scale_limit)
        mask = mask.transpose((2, 0, 1))
        return img, mask, map

    def load_data_sampled(self):
        class_id = np.random.choice(np.arange(0, self.num_classes), p=self.class_probabilities)

        # 2. Randomly select an Image containing this Class (from preprocessing)
        img_file = np.random.choice(self.class_occurrences[class_id])

        # 3. Randomly select a Point in the Image which belongs to the Class (from preprocessing)
        with open(join(self.root, "class_locations", img_file + ".json"), "r") as file:
            data = json.load(file)[str(class_id)]
        pt_idx = np.random.randint(0, len(data["x"]))
        x, y = data["x"][pt_idx], data["y"][pt_idx]

        # 4. Find the index of the selected file in self.mask_files and Load Image and Mask
        idx = [i for i, s in enumerate(self.mask_files) if img_file in s][0]
        img, mask, map = Fishinspector_dataset_PL.load_data(self, idx)

        # 5. Center Crop the image by the selected Point
        mask = mask.transpose((1, 2, 0))
        img, mask = keypoint_scale_crop(img, mask, self.patch_size, (x, y), self.scale_limit)
        mask = mask.transpose((2, 0, 1))

        return img, mask, map


if __name__ == "__main__":
    from tqdm import tqdm

    lateral_classes = [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    root = "/media/l727r/data/UFZ_2023_Fishinspector/Dataset222_Fishinspector"
    dataset = Fishinspector_dataset_PL(
        root=root,
        img_folder="imagesTr",
        num_classes=16,
        label_folder="labelsTr",
        split="train",
        fold="all",
    )
    # img, mask, labeled = dataset[0]
    # print(img.shape, mask.shape, labeled.shape)
    # quit()
    img_files = dataset.img_files
    #
    # num_classes = 16
    # count = [0 for _ in range(0, num_classes)]
    files = []
    for i in range(len(dataset)):
        # for i in range(len(dataset)):
        img, mask, label_map = dataset[i]
        not_empty = np.any(mask == 1, axis=(1, 2)).astype(int)
        if not any(not_empty):
            files.append(img_files[i].rsplit("/", 1)[-1])

    for f in files:
        print(f)
    print(len(files))
    quit()
    # lateral_img = "Lateral" in img_files[i]
    # for j in range(0, num_classes):
    #     if lateral_img and lateral_classes[j]:
    #         if j == 12 and label_map[j] and np.any(mask[j]):
    #             print(img_files[i])
    #         if label_map[j] and not np.any(mask[j]):
    #             count[j] += 1
    #     elif not lateral_img and not lateral_classes[j]:
    #         if label_map[j] and not np.any(mask[j]):
    #             count[j] += 1
    #     if label_map[j] and not np.any(mask[6])
    #
    # if label_map[6] and not np.any(mask[6]) and "Dorsal" in img_files[i]:
    #     print(img_files[i])
    #     count += 1
    for i, c in enumerate(count):

        print(f"{i}: {count[i]}")
