import cv2
import torch
import json
import os
from src.utils import get_logger
from os.path import join
import numpy as np
import random

log = get_logger(__name__)


class Fishinspector_dataset_old(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transforms=None, fold=0):
        self.root = root
        self.img_folder = "imagesTr"
        self.label_folder = "labelsTr"
        self.num_classes = 16

        self.dorsal_classes = torch.tensor(
            [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool
        )
        self.lateral_classes = torch.tensor(
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool
        )
        ignore_files = [
            "Set_2_Dorsal__13112020_Sunitinib_PTK787_0-96hpf",
            "Set_2_Lateral__28052021_SU4312_72-96hpf_tail",
            "Set_3_Lateral__221007-ValproicAcidVAST96hpf_head_W",
            "Set_3_Lateral__221014-DiphenlyamineVAST96hpf_head_W",
        ]

        self.imgs = os.listdir(join(root, self.img_folder))
        self.imgs = [img for img in self.imgs if not any([ignf in img for ignf in ignore_files])]

        seed = 123  # seed is needed since the data should always be shuffled in the same way
        random.Random(seed).shuffle(self.imgs)
        if split == "train":
            self.imgs = self.imgs[0 : int(len(self.imgs) * 0.8)]
        else:
            self.imgs = self.imgs[int(len(self.imgs) * 0.8) :]

        print(f"{len(self.imgs)} found")
        # save the transformations
        self.transforms = transforms

    def __getitem__(self, idx):
        # reading images and masks as numpy arrays
        img = cv2.imread(join(self.root, self.img_folder, self.imgs[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order
        masks = []
        for i in range(0, self.num_classes):
            # mask_name = join(self.root, self.img_folder, self.imgs[idx].replace("_0000", f"_{i:04d}"))
            mask = cv2.imread(
                join(self.root, self.label_folder, self.imgs[idx].replace("_0000", f"_{i:04d}")), -1
            )
            masks.append(mask)
        masks = np.array(masks)
        with open(
            join(self.root, self.label_folder, self.imgs[idx].replace("_0000.png", ".json"))
        ) as file:
            labeled_classes = json.load(file)["annotated_labels"]

            # map = torch.zeros(self.num_classes, dtype=bool)
            if "Dorsal" in self.imgs[idx]:
                map = self.lateral_classes.clone()
            elif "Lateral" in self.imgs[idx]:
                map = self.dorsal_classes.clone()

            map[labeled_classes] = True

        # mask = cv2.imread(self.masks[idx], -1)

        # that's how you apply Albumentations transformations
        if self.transforms is not None:
            masks = masks.transpose((1, 2, 0))
            transformed = self.transforms(image=img, mask=masks)
            img = transformed["image"]  # / 255
            masks = transformed["mask"].permute(2, 0, 1).long()  # type(torch.float16)  # .long()

        # print(masks.shape)
        return img, masks, map  # , self.imgs[idx]

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":

    # quit()
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    root = "/media/l727r/data/UFZ_2023_Fishinspector/Dataset222_Fishinspector"
    dataset = Fishinspector_dataset(root=root, split="train")
    confmatrix = np.zeros((dataset.num_classes, dataset.num_classes))
    count = np.zeros(dataset.num_classes)
    count_p = np.zeros(dataset.num_classes)
    shapes = []
    shapes_count = []
    for i in tqdm(range(len(dataset))):
        img, mask, label_map = dataset[i]

        if img.shape in shapes:
            shapes_count[shapes.index(img.shape)] += 1
        else:
            shapes.append(img.shape)
            shapes_count.append(0)
        count_p += np.any(mask == 1, axis=(1, 2)).astype(int)

        count[label_map] += 1
        for id, j in enumerate(label_map):
            if j:
                confmatrix[id, label_map] += 1
        # break

    print(shapes)
    print(shapes_count)
    print("-----------------")
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
    print("-----------------")
    print(count)
    plt.figure(figsize=(15, 9))
    plt.bar(labels, count, color="#414487FF")
    plt.title("Occurence or Classes", weight="bold")
    plt.ylabel("Number of Occurences", weight="bold")
    plt.xticks(rotation=30)
    plt.xlabel("Classes", weight="bold")
    plt.tight_layout()
    plt.show()
    print("-----------------")

    print(count_p)
    plt.figure(figsize=(15, 9))
    plt.bar(labels, count_p, color="#414487FF")
    plt.title("Not Empty Masks", weight="bold")
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
