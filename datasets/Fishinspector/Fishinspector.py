import cv2
import torch
import json
import os
from src.utils import get_logger
from os.path import join
import numpy as np
import random
from datasets.Base_Datasets.multilabel import Multilabel_Dataset, Multilabel_CV_Dataset

log = get_logger(__name__)


class Fishinspector_dataset(Multilabel_CV_Dataset):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes

        self.ignore_files = [
            "Set_2_Dorsal__13112020_Sunitinib_PTK787_0-96hpf",
            "Set_2_Lateral__28052021_SU4312_72-96hpf_tail",
            "Set_3_Lateral__221007-ValproicAcidVAST96hpf_head_W",
            "Set_3_Lateral__221014-DiphenlyamineVAST96hpf_head_W",
        ]

        super().__init__(**kwargs)

    def get_img_files(self) -> list:
        img_files = super().get_img_files()
        img_files = [
            img for img in img_files if not any([ignf in img for ignf in self.ignore_files])
        ]
        return img_files

    def get_mask_files(self) -> list:
        return self.img_files

    def load_mask(self, idx):
        masks = []
        for i in range(0, self.num_classes):
            img_name = os.path.split(self.img_files[idx])[-1]
            mask = cv2.imread(
                join(self.root, self.label_folder, img_name.replace("_0000", f"_{i:04d}")), -1
            )
            masks.append(mask)
        return np.array(masks)


class Fishinspector_dataset_partly_labeled(Fishinspector_dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dorsal_classes = torch.tensor(
            [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool
        )
        self.lateral_classes = torch.tensor(
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool
        )

    def __getitem__(self, idx):

        img, mask = super().__getitem__(idx)

        file_name = os.path.split(self.img_files[idx])[-1].replace("_0000.png", ".json")

        with open(join(self.root, self.label_folder, file_name)) as file:
            labeled_classes = json.load(file)["annotated_labels"]

            if "Dorsal" in file_name:
                map = self.lateral_classes.clone()
            elif "Lateral" in file_name:
                map = self.dorsal_classes.clone()
            map[labeled_classes] = True

        return img, mask, map

        # # reading images and masks as numpy arrays
        # img = cv2.imread(join(self.root, self.img_folder, self.imgs[idx]))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order
        # masks = []
        # for i in range(0, self.num_classes):
        #     # mask_name = join(self.root, self.img_folder, self.imgs[idx].replace("_0000", f"_{i:04d}"))
        #     mask = cv2.imread(
        #         join(self.root, self.label_folder, self.imgs[idx].replace("_0000", f"_{i:04d}")), -1
        #     )
        #     masks.append(mask)
        # masks = np.array(masks)
        # with open(
        #     join(self.root, self.label_folder, self.imgs[idx].replace("_0000.png", ".json"))
        # ) as file:
        #     labeled_classes = json.load(file)["annotated_labels"]
        #
        #     map = torch.zeros(self.num_classes, dtype=bool)
        #     if "Dorsal" in self.imgs[idx]:
        #         map = self.lateral_classes.clone()
        #     elif "Lateral" in self.imgs[idx]:
        #         map = self.dorsal_classes.clone()
        #
        #     map[labeled_classes] = True
        #
        # # mask = cv2.imread(self.masks[idx], -1)
        #
        # # thats how you apply Albumentations transformations
        # if self.transforms is not None:
        #     masks = masks.transpose((1, 2, 0))
        #     transformed = self.transforms(image=img, mask=masks)
        #     img = transformed["image"]  # / 255
        #     masks = transformed["mask"].permute(2, 0, 1).long()  # type(torch.float16)  # .long()
        #
        # # print(masks.shape)
        # return img, masks, map  # , self.imgs[idx]


if __name__ == "__main__":

    # quit()
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    root = "/media/l727r/data/UFZ_2023_Fishinspector/Dataset222_Fishinspector"
    dataset = Fishinspector_dataset(root=root, split="train")
    confmatrix = np.zeros((dataset.num_classes, dataset.num_classes))
    count = np.zeros(dataset.num_classes)
    for i in tqdm(range(len(dataset))):
        img, mask, label_map = dataset[i]
        count[label_map] += 1
        for id, j in enumerate(label_map):
            if j:
                confmatrix[id, label_map] += 1

    print(confmatrix)

    # labels = cfg.DATASET.CLASS_LABELS
    # count_norm = np.array(confmatrix).astype("float") / confmatrix.diag()[:, np.newaxis]
    labels = [
        "contour_LAT",
        "yolk_DV",
        "mouth_tip_LAT",
        "eye_LAT",
        "pericard_LAT",
        "eye1_DV",
        "contour_DV",
        "fin1_DV",
        "otolith1_LAT",
        "otolith2_LAT",
        "eye2_DV",
        "notochord_LAT",
        "swimbladder_LAT",
        "yolk_LAT",
        "fin2_DV",
        "pigmentation_LAT",
    ]
    plt.figure(figsize=(9, 9))
    # confmatrix = np.array(confmatrix).astype("float") / np.array(confmatrix.diag()[:, np.newaxis]
    plt.imshow(confmatrix, interpolation="nearest", cmap=plt.cm.viridis)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=-90)
    plt.yticks(tick_marks, labels)
    plt.colorbar()
    plt.title("Occurrences of Class A together with Class B", weight="bold")
    plt.ylabel("Classes A", weight="bold")
    plt.xlabel("Classes B", weight="bold")
    plt.tight_layout()

    # plt.figure(figsize=(9, 9))
    # confmatrix = np.array(confmatrix).astype("float") / confmatrix.diagonal()[:, np.newaxis]
    # plt.imshow(confmatrix, interpolation="nearest", cmap=plt.cm.viridis)
    # tick_marks = np.arange(len(labels))
    # plt.xticks(tick_marks, labels, rotation=-90)
    # plt.yticks(tick_marks, labels)
    # plt.colorbar()
    # plt.title("Probability of Class A to appear together with Class B", weight="bold")
    # plt.ylabel("Classes A", weight="bold")
    # plt.xlabel("Classes B", weight="bold")
    # plt.tight_layout()
    print(count)
    plt.figure(figsize=(15, 9))
    plt.bar(labels, count, color="#414487FF")
    plt.title("Occurence or Classes", weight="bold")
    plt.ylabel("Number of Occurences", weight="bold")
    plt.xticks(rotation=30)
    plt.xlabel("Classes", weight="bold")
    plt.tight_layout()
    plt.show()

    # Set_2_Dorsal__13112020_Sunitinib_PTK787_0-96hpf [970 - 1031] translation horizontal
    # Set_2_Lateral__28052021_SU4312_72-96hpf_tail [3692 - 3735] mirror horizontal
    # Set_3_Lateral__221007-ValproicAcidVAST96hpf_head_W [7274 - 7290] mirror horizontal
    # Set_3_Lateral__221014-DiphenlyamineVAST96hpf_head_W [7314 - 7338] mirror horizontal

    # Set_2_Lateral__17092021_Thalidomide_Dech_24-96hpf_tail_W_B11_1_4_0000.png
    # Set_2_Lateral__17092021_Thalidomide_Dech_24-96hpf_tail_W_B07_1_4_0000.png
    # Set_3_Lateral__221014-NiflumicAcidVAST96hpf_head_W_H11_1_2_0000.png
    # Set_3_Dorsal__220304_Butoxyethanol_96hpf001_tail_W_C01_1_3_0000.png
    # Set_2_Lateral__25062021_SU4312_24-96hpf_Replikat2_tail_W_A01_1_4_0000.png
