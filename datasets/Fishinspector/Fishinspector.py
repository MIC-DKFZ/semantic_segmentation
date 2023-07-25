import glob

import cv2
import os
from src.utils import get_logger
from os.path import join
import numpy as np
from src.dataset_utils import split_by_ID
from datasets.Base_Datasets.base import Base_Dataset
from datasets.Base_Datasets.multilabel import (
    Multilabel_CV_Dataset,
    Multilabel_Sampling_CV_Dataset,
    Multilabel_Sampling_Dataset,
)

cv2.setNumThreads(0)
log = get_logger(__name__)


class Fishinspector_Base(Base_Dataset):
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

        img_files = super().get_img_files()

        mask_files = [
            join(self.root, self.label_folder, os.path.split(img_file)[-1])
            for img_file in img_files
        ]

        mask_files = [
            mask for mask in mask_files if not any([ignf in mask for ignf in self.ignore_files])
        ]
        return mask_files

    def load_mask(self, idx):
        masks = []
        mask_file = self.mask_files[idx]
        for i in range(0, self.num_classes):
            mask = cv2.imread(mask_file.replace("_0000", f"_{i:04d}"), -1)
            masks.append(mask)
        return np.array(masks, dtype=np.uint8)

    def get_split_ids(self):
        files = [os.path.split(file)[-1] for file in self.img_files]
        IDs_S1 = [file.split("--", 1)[0] for file in files if "Set_1_" in file]
        IDs_S2 = [file.split("_W_", 1)[0] for file in files if "Set_2_" in file]
        files_S3 = [file for file in files if "Set_3_" in file]
        IDs_S3 = [
            file.split("_W_", 1)[0].split("_controlplateW", 1)[0]
            for file in files_S3
            if "_controlplateW" in file or "_W_" in file
        ]
        IDs_S3 += [
            file.split("--")[0]
            for file in files_S3
            if not "_controlplateW" in file and not "_W_" in file
        ]
        IDs = np.unique(IDs_S1 + IDs_S2 + IDs_S3)
        IDs = [ID for ID in IDs if not any(JD in ID and JD != ID for JD in IDs)]
        return IDs


class Fishinspector_dataset(Multilabel_CV_Dataset, Fishinspector_Base):
    pass


class Fishinspector_sampling_dataset(Multilabel_Sampling_Dataset, Fishinspector_Base):
    pass


if __name__ == "__main__":
    print(Fishinspector_sampling_dataset.mro())
    root = "/media/l727r/data/UFZ_2023_Fishinspector/Dataset222_Fishinspector"
    dataset = Fishinspector_sampling_dataset(
        root=root,
        img_folder="imagesTr",
        num_classes=16,
        label_folder="labelsTr",
        split="train",
        fold="all",
    )
    img, mask = dataset[0]
    print(img.shape, mask.shape)
    quit()

    root = "/media/l727r/data/UFZ_2023_Fishinspector/Dataset222_Fishinspector"

    files = glob.glob(join(root, "imagesTr", "*.png"))
    files = [os.path.split(file)[-1] for file in files]
    # files = [
    #     file.split("_W_", 1)[0] for file in files if "Set_3_Dorsal__2022_02_11_11_39_47" in file
    # ]
    # print(np.unique(files))
    # quit()

    IDs_S1 = [file.split("--", 1)[0] for file in files if "Set_1_" in file]
    IDs_S2 = [file.split("_W_", 1)[0] for file in files if "Set_2_" in file]
    files_S3 = [file for file in files if "Set_3_" in file]
    IDs_S3 = [
        file.split("_W_", 1)[0].split("_controlplateW", 1)[0]
        for file in files_S3
        if "_controlplateW" in file or "_W_" in file
    ]
    IDs_S3 += [
        file.split("--")[0]
        for file in files_S3
        if not "_controlplateW" in file and not "_W_" in file
    ]
    IDs = np.unique(IDs_S1 + IDs_S2 + IDs_S3)
    IDs = [ID for ID in IDs if not any(JD in ID and JD != ID for JD in IDs)]

    print(len(IDs))
    for id in IDs:
        for jd in IDs:
            if id in jd and id != jd:
                print("I", id)
                print("J", jd)

    split_by_ID(files, IDs)

    # files_S2 = [file for file in files if "Set_3_" in file]
    #
    # IDs_S3_1 = [
    #     file.split("_W_", 1)[0].split("_controlplateW", 1)[0]
    #     for file in files_S2
    #     if "_controlplateW" in file or "_W_" in file
    # ]
    # IDs_S3_2 = [
    #     file.split("--")[0]
    #     for file in files_S2
    #     if not "_controlplateW" in file and not "_W_" in file
    # ]
    # IDs_S3 = IDs_S3_1 + IDs_S3_2
    # # files_ID = [file.split("_W_", 1)[0].split("_controlplateW", 1)[0] for file in files_S2]
    # # files_ID = [file.split("_W_", 1)[0].split("controlplateW_", 1)[0] for file in files_S2]
    # IDs, count = np.unique(IDs_S3, return_counts=True)
    # # print(files)
    # for i, c in zip(IDs, count):
    #     print(i, c)
    # print(len(files_S2), len(IDs))

    quit()
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
