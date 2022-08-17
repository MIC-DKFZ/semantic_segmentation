import argparse
import os
import sys
from itertools import combinations, combinations_with_replacement
import omegaconf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import hydra
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils import has_not_empty_attr


def get_mask_stats(dataloader: DataLoader, num_classes: int):
    pixel_count = torch.zeros(num_classes, dtype=int)
    sample_count = torch.zeros((num_classes, num_classes), dtype=int)
    ignore = 0
    for _, mask in tqdm(dataloader):
        mask = np.array(mask)
        val, cnt = np.unique(mask, return_counts=True)
        for v, c in zip(val, cnt):
            if v >= 0 and v < num_classes:
                pixel_count[v] += c
            else:
                ignore += c

        for m in mask:
            val_m = [v_m for v_m in val if v_m in m and v_m >= 0 and v_m < num_classes]
            for a, b in combinations_with_replacement(val_m, 2):
                sample_count[a, b] += 1
                if a != b:
                    sample_count[b, a] += 1

    return pixel_count, sample_count, ignore


def get_mean_std_global(dataloader: DataLoader) -> tuple:
    """
    Computing the mean and std for each channel over the dataset
    This is done globaly over all pixels in the dataset
    only works for small datasets
    memory problems for larger datasets -> use get_mean_std_sample instead

    Parameters
    ----------
    dataloader : DataLoader

    Returns
    -------
    tuple : (list, list)
        mean and std for each channel as a list
    """
    elements = None
    for img, mask in tqdm(dataloader):
        img = img.permute(1, 0, 2, 3)
        img = img.flatten(start_dim=1)

        if elements == None:
            elements = img
        else:
            elements = torch.cat((elements, img), dim=1)
    mean = elements.mean(1)
    std = elements.std(1)
    print("Mean per Channel:", mean.tolist())
    print("STD per Channel:", std.tolist())
    return mean, std


def get_mean_std_sample(dataloader: DataLoader) -> tuple:
    """
    Computing the mean and std for each channel over the dataset
    This is done sample-wise over all samples in the dataset
    https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c

    Parameters
    ----------
    dataloader : DataLoader

    Returns
    -------
    tuple : (list, list)
        mean and std for each channel as a list
    """
    elements = None
    elements_squared = None
    num_batches = 0
    for img, mask in tqdm(dataloader):
        img = img.permute(1, 0, 2, 3)

        img_squared = torch.mean(img**2, dim=[1, 2, 3])
        img = torch.mean(img, dim=[1, 2, 3])
        num_batches += 1

        if elements is None:
            elements = img
            elements_squared = img_squared
        else:
            elements += img
            elements_squared += img_squared
    mean = elements / num_batches
    std = (elements_squared / num_batches - mean**2) ** 0.5
    return mean, std


def get_dataset_stats(
    overrides_cl: list,
    name: str = None,
    split: str = "train",
    img_only: bool = False,
    mask_only: bool = False,
    supress_output: bool = False,
) -> None:
    """
    Computing the mean and std of each channel over all images inside the datasets
    Computing weights for each Class in the Dataset
    https://github.com/openseg-group/OCNet.pytorch/issues/14#issuecomment-451134346

    Parameters
    ----------
    datasets : list
        List of torch Datasets over which the stats are computed
    num_classes : int
        number of classes in the dataset
    input_channels : int, optional
        number of input channels in the dataset
    """
    hydra.initialize(config_path="../config")
    cfg = hydra.compose(config_name="baseline", overrides=overrides_cl)

    num_classes = cfg.DATASET.NUM_CLASSES
    input_channels = cfg.input_channels if has_not_empty_attr(cfg, "input_channels") else 3

    transforms = A.Compose(
        [A.Normalize(mean=[0] * input_channels, std=[1] * input_channels), ToTensorV2()]
    )

    dataset = hydra.utils.instantiate(cfg.dataset, split=split, transforms=transforms)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    if not mask_only:
        # Extracting mean and std for each channel from the Image Data
        print("Extracting Image Information...")
        mean, std = get_mean_std_sample(dataloader)
        print("Mean per Channel: {}".format(mean.tolist()))
        print("STD per Channel: {}".format(std.tolist()))
    if not img_only:
        # Counting Pixels and Occurrences for each Class from the Label Data
        print("Extracting Label Information...")
        count, sample_count, ignore = get_mask_stats(dataloader, num_classes)
        print("Pixel per Class (with  {} pixels ignored):\n{}".format(ignore, count.tolist()))
        print(
            "Images Containing each Class (total number of Images: {}):\n{}".format(
                len(dataset), list(sample_count.diag().tolist())
            )
        )

        # Computing Weights
        weights_inv = 1 - count / count.sum()
        print("Inverse Class Weights:\n{}".format(weights_inv.tolist()))
        weights_bal = (1 / count) * (count.sum() / num_classes)
        print("Balanced Class Weights:\n{} ".format(weights_bal.tolist()))

    if not supress_output:
        """
        Define Output Dir
        """
        output_dir = os.path.join(os.getcwd(), "dataset_stats", cfg.DATASET.NAME)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        """
        Save Stats to txt file
        """
        if name is not None:
            file_name = os.path.join(output_dir, name + "_dataset_stats.txt")
        else:
            file_name = os.path.join(output_dir, "dataset_stats.txt")
        with open(file_name, "a") as f:
            if not mask_only:
                f.write("Mean per Channel: {} \n".format(mean.tolist()))
                f.write("STD per Channel: {} \n".format(std.tolist()))
            if not img_only:
                f.write(
                    "Pixel per Class (with  {} pixels ignored):\n{} \n".format(
                        ignore, count.tolist()
                    )
                )
                f.write(
                    "Images Containing each Class (total number of Images: {}):\n{} \n".format(
                        len(dataset), sample_count.diag().tolist()
                    )
                )
                f.write(
                    "Average size per Class in pixel:\n{} \n".format(
                        (count / sample_count.diag()).tolist()
                    )
                )

                f.write("Inverse Class Weights:\n{} \n".format(weights_inv.tolist()))
                f.write("Balanced Class Weights:\n{} \n".format(weights_bal.tolist()))

        if not img_only:
            """
            Number of pixel of each class in the dataset
            Color Viridis: https://rpubs.com/mjvoss/psc_viridisplt
            """
            pixel_labels = cfg.DATASET.CLASS_LABELS.copy()
            if ignore == 0:
                pixel_count = np.array(count)
            else:
                pixel_labels.append("ignored")
                pixel_count = np.array(torch.cat((count, torch.tensor([ignore]))))
            plt.figure(figsize=(15, 9))
            plt.bar(pixel_labels, pixel_count, color="#414487FF")
            plt.title("Number of pixel per class", weight="bold")
            plt.ylabel("Number of pixel", weight="bold")
            plt.xticks(rotation=30)
            plt.xlabel("Classes", weight="bold")
            plt.tight_layout()
            if name != None:
                plot_name = name + "_Pixel_per_Class.png"
            else:
                plot_name = "Pixel_per_Class.png"
            output_name = os.path.join(output_dir, plot_name)
            plt.savefig(output_name)
            # plt.show()

            """
            Number of occurrences of each class
            Color Viridis: https://rpubs.com/mjvoss/psc_viridisplt
            """
            labels = cfg.DATASET.CLASS_LABELS
            plt.figure(figsize=(15, 9))
            plt.bar(labels, sample_count.diag(), color="#414487FF")
            plt.hlines(len(dataset), 0 - 0.5, num_classes - 0.5, "red")
            plt.text(
                num_classes - 0.5,
                len(dataset),
                "total number of images",
                ha="right",
                va="bottom",
                fontsize="large",
                color="red",
            )
            plt.title("Number of occurrences per class", weight="bold")
            plt.ylabel("Number of occurrences", weight="bold")
            plt.xticks(rotation=30)
            plt.xlabel("Classes", weight="bold")
            plt.tight_layout()
            if name is not None:
                plot_name = name + "_Class_Occurrences.png"
            else:
                plot_name = "Class_Occurrences.png"
            output_name = os.path.join(output_dir, plot_name)
            plt.savefig(output_name)
            # plt.show()

            """
            Average size of each class
            Color Viridis: https://rpubs.com/mjvoss/psc_viridisplt
            """
            labels = cfg.DATASET.CLASS_LABELS
            plt.figure(figsize=(15, 9))
            plt.bar(labels, count / sample_count.diag(), color="#414487FF")
            plt.title("Average Size of Classes (in pixel)", weight="bold")
            plt.ylabel("Number of pixel", weight="bold")
            plt.xticks(rotation=30)
            plt.xlabel("Classes", weight="bold")
            plt.tight_layout()

            if name is not None:
                plot_name = name + "_Average_Class_Size.png"
            else:
                plot_name = "Average_Class_Size.png"
            output_name = os.path.join(output_dir, plot_name)
            plt.savefig(output_name)
            # plt.show()

            """
            Number of co-occurrences of each class-pair
            Probability of Class A to appear together with Class B
            """
            labels = cfg.DATASET.CLASS_LABELS
            count_norm = np.array(sample_count).astype("float") / sample_count.diag()[:, np.newaxis]

            plt.figure(figsize=(9, 9))
            plt.imshow(count_norm, interpolation="nearest", cmap=plt.cm.viridis)
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)
            plt.colorbar()
            plt.title("Probability of Class A to appear together with Class B", weight="bold")
            plt.ylabel("Classes A", weight="bold")
            plt.xlabel("Classes B", weight="bold")
            plt.tight_layout()

            if name is not None:
                plot_name = name + "_Probability_of_Co_Occurrence.png"
            else:
                plot_name = "Probability_of_Co_Occurrence.png"
            output_name = os.path.join(output_dir, plot_name)
            plt.savefig(output_name)
            # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the sub folder to save the results in",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="which split to use: train, val or test",
    )
    parser.add_argument("--img_only", action="store_true", help="Only analyse Image Data")
    parser.add_argument("--mask_only", action="store_true", help="Only analyse Mask Data")
    parser.add_argument(
        "--supress_output",
        action="store_true",
        help="Supress creation of a output directory and output files",
    )
    args, overrides = parser.parse_known_args()

    name = args.name
    split = args.split
    img_only = args.img_only
    mask_only = args.mask_only
    supress_output = args.supress_output
    get_dataset_stats(overrides, name, split, img_only, mask_only, supress_output)
