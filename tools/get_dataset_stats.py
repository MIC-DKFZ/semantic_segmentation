import argparse
import os
import sys
from itertools import combinations_with_replacement
import ruamel.yaml as ryaml

# import yaml
from omegaconf import OmegaConf
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import hydra
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.utils import has_not_empty_attr
import cv2
from matplotlib import cm


def process_batch(dataset, index, segmentation):
    img, mask = dataset[index]
    result = {}
    result["mean"] = torch.mean(img, dim=[1, 2]).tolist()
    result["std"] = torch.std(img, dim=[1, 2]).tolist()
    result["size"] = img.shape[1:]
    if segmentation == "semantic":
        val, cnt = np.unique(mask, return_counts=True)
    elif segmentation == "multilabel":
        res = np.sum(np.array(mask), axis=(1, 2))
        non_zero_indices = np.nonzero(res)[0]
        val = []
        cnt = []
        for i in non_zero_indices:
            val.append(i)
            cnt.append(res[i])
        val = np.array(val, dtype=int)
        cnt = np.array(cnt, dtype=int)
    else:
        print(f"segmentation type: {segmentation} is not supported, no mask stats will be provided")
        val = []
        cnt = []

    result["class_occ"] = val
    result["class_size"] = cnt

    return result


def process_dataset(dataset, num_classes, segmentation="semantic", num_processes=8):

    results = {"mean": [], "std": [], "size": [], "class_occ": [], "class_size": []}
    pool = multiprocessing.Pool(processes=num_processes)
    # Get Image and Mask Properties
    results_list = []
    for i in tqdm(range(0, len(dataset)), desc="Initializing"):
        # results_list.append(process_batch(dataset, i, segmentation))
        results_list.append(
            pool.starmap_async(
                process_batch,
                (
                    (
                        dataset,
                        i,
                        segmentation,
                    ),
                ),
            )
        )
    results_list = [a.get() for a in tqdm(results_list, desc="Processing")]
    pool.close()
    pool.join()
    for r in tqdm(results_list, desc="Aggregating"):
        results["mean"].append(r[0]["mean"])
        results["std"].append(r[0]["std"])
        results["size"].append(r[0]["size"])
        results["class_occ"].append(r[0]["class_occ"])
        results["class_size"].append(r[0]["class_size"])

    # Convert Everything into the right format and aggregate the properties
    output = OmegaConf.create(
        {
            "mean": [],
            "std": [],
            "img_size": {"shapes": [], "min": None, "max": None, "mean": None},
            "class_occurrence": None,
            "class_size": None,
        }
    )
    output.mean = np.mean(np.array(results["mean"]), axis=0).tolist()
    output.std = np.mean(np.array(results["std"]), axis=0).tolist()

    tuple_dtype = np.dtype([("x", int), ("y", int)])
    shapes = np.unique(np.array(results["size"], dtype=tuple_dtype))
    output.img_size.shapes = shapes.tolist()
    output.img_size.mean = np.round(np.mean(results["size"], axis=0)).astype(int).tolist()
    output.img_size.min = np.min(results["size"], axis=0).tolist()
    output.img_size.max = np.max(results["size"], axis=0).tolist()

    occurances = np.zeros(num_classes)
    sizes = np.zeros(num_classes)
    for occ, size in zip(results["class_occ"], results["class_size"]):
        for o, s in zip(occ, size):
            if o >= 0 and o < num_classes:
                occurances[o] += 1
                sizes[o] += s
    output.class_occurrence = occurances.tolist()
    output.class_size = (np.round(sizes / occurances)).tolist()
    return output


def process_dataset2(dataset, num_classes, segmentation="semantic"):

    results = {"mean": [], "std": [], "size": [], "class_occ": [], "class_size": []}

    # Get Image and Mask Properties
    for img, mask in tqdm(dataset):

        results["mean"].append(torch.mean(img, dim=[1, 2]).tolist())
        results["std"].append(torch.std(img, dim=[1, 2]).tolist())
        results["size"].append(img.shape[1:])
        if segmentation == "semantic":
            val, cnt = np.unique(mask, return_counts=True)
        elif segmentation == "multilabel":
            result = np.sum(np.array(mask), axis=(1, 2))
            non_zero_indices = np.nonzero(result)[0]
            val = []
            cnt = []
            for i in non_zero_indices:
                val.append(i)
                cnt.append(result[i])
        else:
            print(
                f"segmentation type: {segmentation} is not supported, no mask stats will be"
                " provided"
            )
            val = []
            cnt = []
        results["class_occ"].append(val)
        results["class_size"].append(cnt)

    # Convert Everything into the right format and aggregate the properties
    output = OmegaConf.create(
        {
            "mean": [],
            "std": [],
            "img_size": {"shapes": [], "min": None, "max": None, "mean": None},
            "class_occurrence": None,
            "class_size": None,
        }
    )
    output.mean = np.mean(np.array(results["mean"]), axis=0).tolist()
    output.std = np.mean(np.array(results["std"]), axis=0).tolist()

    tuple_dtype = np.dtype([("x", int), ("y", int)])
    shapes = np.unique(np.array(results["size"], dtype=tuple_dtype))
    output.img_size.shapes = shapes.tolist()
    output.img_size.mean = np.round(np.mean(results["size"], axis=0)).astype(int).tolist()
    output.img_size.min = np.min(results["size"], axis=0).tolist()
    output.img_size.max = np.max(results["size"], axis=0).tolist()

    occurances = np.zeros(num_classes)
    sizes = np.zeros(num_classes)
    for occ, size in zip(results["class_occ"], results["class_size"]):
        for o, s in zip(occ, size):
            if o >= 0 and o < num_classes:
                occurances[o] += 1
                sizes[o] += s
    output.class_occurrence = occurances.tolist()
    output.class_size = (np.round(sizes / occurances)).tolist()
    return output


def get_dataset_stats(
    overrides_cl: list,
    name: str = None,
    split: str = "train",
    img_only: bool = False,
    mask_only: bool = False,
    supress_output: bool = False,
) -> None:

    hydra.initialize(config_path="../config", version_base="1.3")
    cfg = hydra.compose(config_name="training", overrides=overrides_cl)

    output = OmegaConf.create()

    """
    General Information about the Dataset
    """
    output.Info = {
        "name": None,
        "size": {},
        "num_classes": None,
        "ignore_index": None,
        "class_labels": None,
        "segmentation_type": None,
    }
    output.Info.name = cfg.dataset.name
    output.Info.num_classes = cfg.dataset.num_classes
    output.Info.ignore_index = cfg.dataset.ignore_index
    output.Info.class_labels = cfg.dataset.class_labels
    output.Info.segmentation_type = cfg.dataset.segmentation_type

    input_channels = cfg.input_channels if has_not_empty_attr(cfg, "input_channels") else 3
    transforms = A.Compose(
        [A.Normalize(mean=[0] * input_channels, std=[1] * input_channels), ToTensorV2()]
    )

    dataset_train = hydra.utils.instantiate(
        cfg.dataclass, split="train", transforms=transforms, _recursive_=False
    )
    dataset_val = hydra.utils.instantiate(
        cfg.dataclass, split="val", transforms=transforms, _recursive_=False
    )
    # dataset_test = hydra.utils.instantiate(cfg.dataset.dataset, split="test", transforms=transforms)

    output.Info.size.train = len(dataset_train)
    output.Info.size.val = len(dataset_val)
    # output.Info.size.test = len(dataset_test)
    """
    Information about the Train Set
    """
    # output.Train = {}
    output.Train = process_dataset(
        dataset_train, cfg.dataset.num_classes, cfg.dataset.segmentation_type, cfg.num_workers
    )
    output.Val = process_dataset(
        dataset_val, cfg.dataset.num_classes, cfg.dataset.segmentation_type, cfg.num_workers
    )

    output.All = OmegaConf.create(
        {
            "mean": [],
            "std": [],
            "img_size": {"min": None, "max": None, "mean": None},
            "class_occurrence": None,
            "class_size": None,
        }
    )
    output.All.mean = (
        (
            np.array(output.Train.mean) * len(dataset_train)
            + np.array(output.Val.mean) * len(dataset_val)
        )
        / (len(dataset_val) + len(dataset_train))
    ).tolist()

    output.All.std = (
        (
            np.array(output.Train.std) * len(dataset_train)
            + np.array(output.Val.std) * len(dataset_val)
        )
        / (len(dataset_val) + len(dataset_train))
    ).tolist()

    output.All.img_size.mean = (
        (
            np.array(output.Train.img_size.mean) * len(dataset_train)
            + np.array(output.Val.img_size.mean) * len(dataset_val)
        )
        / (len(dataset_val) + len(dataset_train))
    ).tolist()

    output.All.img_size.min = (
        np.min(np.array([output.Train.img_size.min, output.Val.img_size.min]), axis=0)
    ).tolist()

    output.All.img_size.max = (
        np.max(np.array([output.Train.img_size.max, output.Val.img_size.max]), axis=0)
    ).tolist()

    output.All.class_occurrence = (
        np.array(output.Train.class_occurrence) + np.array(output.Val.class_occurrence)
    ).tolist()

    output.All.class_size = (
        (
            np.array(output.Train.class_size) * np.array(output.Train.class_occurrence)
            + np.array(output.Val.class_size) * np.array(output.Val.class_occurrence)
        )
        / np.array(output.All.class_occurrence)
    ).tolist()

    # Save as Yaml File
    yaml = ryaml.YAML()
    yaml.width = 120  # Adjust this to your desired line width
    yaml.default_flow_style = None  # Disable the default flow style

    os.makedirs("ds_stats", exist_ok=True)
    with open(f"../dataset_stats/{cfg.dataset.name}.yaml", "w") as file:
        yaml.dump(OmegaConf.to_container(output), file)  # , default_flow_style=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Prefix for naming the results",
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
