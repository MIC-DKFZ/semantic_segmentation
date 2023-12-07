import numpy as np
import cv2
from src.data_handler.label_handler_multilabel import MultiLabelSegmentationHandler
import torch
import json
import os
from numpy import ndarray


class MultiLabelSegmentationHandlerFI(MultiLabelSegmentationHandler):
    def load_file(self, file: str) -> ndarray:
        masks = []
        file = file.replace("_0000.png", "")
        for i in range(0, self.num_classes):
            mask = cv2.imread(f"{file}_{i:04d}.png", -1)
            masks.append(mask)
        return np.array(masks, dtype=np.uint8)


class MultiLabelSegmentationHandlerFI_PL_v2(MultiLabelSegmentationHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dorsal_classes = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
        self.lateral_classes = np.array(
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool
        )
        self.trusted_classes = np.array(
            [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=bool
        )

    def load_file(self, file: str) -> ndarray:
        masks = []
        file = file.replace("_0000.png", "")
        for i in range(0, self.num_classes):
            mask = cv2.imread(f"{file}_{i:04d}.png", -1)
            masks.append(mask)
        masks = np.array(masks, dtype=np.uint8)

        with open(file + ".json") as f:
            labeled_classes = json.load(f)["annotated_labels"]
            # print(labeled_classes)
            map_labeled_classes = np.zeros(self.num_classes, dtype=bool)
            map_labeled_classes[labeled_classes] = True
        # 2) Get all empty masks
        map_empty_classes = ~np.any(masks == 1, axis=(1, 2)).astype(bool)

        # 3) Get all classes which are not trusted and marked as labeled but have an empty mask
        map_trust_issue = ~self.trusted_classes & map_labeled_classes & map_empty_classes
        # 4) Get all classes of the other set
        if "Dorsal" in os.path.split(file)[-1]:
            map_other_set = self.lateral_classes.copy()
        elif "Lateral" in os.path.split(file)[-1]:
            map_other_set = self.dorsal_classes.copy()

        # 5) Combine: marked as labeled or in the other set but have no trust issue
        map = (map_labeled_classes | map_other_set) & ~map_trust_issue
        map = torch.tensor(map)
        # 6) Empty all Masks of the other set which are not empty
        masks[~map] = -1

        return masks


# class MultiLabelSegmentationHandlerFI_PL(MultiLabelSegmentationHandler):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.dorsal_classes = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
#         self.lateral_classes = np.array(
#             [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool
#         )
#         self.trusted_classes = np.array(
#             [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=bool
#         )
#
#     def apply_transforms(self, img, mask, transforms, *args, **kwargs):
#         mask, map = mask
#         if transforms is not None:
#             mask = mask.transpose((1, 2, 0))
#             transformed = transforms(image=img, mask=mask, *args, **kwargs)
#             img = transformed["image"]
#             mask = transformed["mask"].permute(2, 0, 1)
#         return img, {"mask": mask, "labeled": map}
#
#     def load_mask(self, file):
#         masks = []
#         for i in range(0, self.num_classes):
#             mask = cv2.imread(f"{file}_{i:04d}.png", -1)
#             masks.append(mask)
#         masks = np.array(masks, dtype=np.uint8)
#
#         with open(file + ".json") as f:
#             labeled_classes = json.load(f)["annotated_labels"]
#             # print(labeled_classes)
#             map_labeled_classes = np.zeros(self.num_classes, dtype=bool)
#             map_labeled_classes[labeled_classes] = True
#         # print(sorted(labeled_classes))
#         # 2) Get all empty masks
#         map_empty_classes = ~np.any(masks == 1, axis=(1, 2)).astype(bool)
#
#         # 3) Get all classes which are not trusted and marked as labeled but have an empty mask
#         map_trust_issue = ~self.trusted_classes & map_labeled_classes & map_empty_classes
#         # 4) Get all classes of the other set
#         if "Dorsal" in os.path.split(file)[-1]:
#             map_other_set = self.lateral_classes.copy()
#         elif "Lateral" in os.path.split(file)[-1]:
#             map_other_set = self.dorsal_classes.copy()
#
#         # 5) Combine: marked as labeled or in the other set but have no trust issue
#         map = (map_labeled_classes | map_other_set) & ~map_trust_issue
#         map = torch.tensor(map)
#         # 6) Empty all Masks of the other set which are not empty
#         masks[~map_empty_classes & map_other_set] = np.zeros(masks[0].shape)
#
#         return masks, map
