import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import has_not_empty_attr


def get_class_weights(dataloader: DataLoader, num_classes: int) -> None:
    """
    compute weights (inverted and balanced) for each class in the Dataset

    Parameters
    ----------
    dataloader : DataLoader
    num_classes : int
    """
    count = torch.zeros(num_classes)
    no = 0
    for img, mask in tqdm(dataloader):
        val, coun = np.unique(np.array(mask), return_counts=True)
        for v, c in zip(val, coun):
            if v < num_classes:
                count[v] += c
            else:
                no += c
    print("Total number of pixel per class: ", np.array(count))
    print("      number of ignored pixel: ", no)
    counts = 1 - count / count.sum()
    print("Inverse Weights", np.array(counts))
    counts_bal = (1 / count) * (count.sum() / num_classes)
    print("Balanced Weights: ", np.array(counts_bal))


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
    print("Mean per Channel:", mean.tolist())
    print("STD per Channel:", std.tolist())
    return mean, std


def get_dataset_stats(overrides_cl: list) -> None:
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

    dataset = hydra.utils.instantiate(cfg.dataset, split="train", transforms=transforms)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    # print("## Compute Mean and Std for each Channel")
    # get_mean_std_global(dataloader)
    # get_mean_std_sample(dataloader)

    # print("## Compute Class Weights for each Class")
    get_class_weights(dataloader, num_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args, overrides = parser.parse_known_args()

    get_dataset_stats(overrides)
