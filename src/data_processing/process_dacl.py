"""
Partly adapted from:
https://github.com/phiyodr/dacl10k-toolkit/blob/master/dacl10k/dacl10kdataset.py
"""
import os
import shutil
import multiprocessing
import json
from os.path import join, split

import cv2
import numpy as np
from PIL import Image, ImageDraw

TARGET_LIST = [
    "Crack",
    "ACrack",
    "Wetspot",
    "Efflorescence",
    "Rust",
    "Rockpocket",
    "Hollowareas",
    "Cavity",
    "Spalling",
    "Graffiti",
    "Weathering",
    "Restformwork",
    "ExposedRebars",
    "Bearing",
    "EJoint",
    "Drainage",
    "PEquipment",
    "JTape",
    "WConccor",
]
target_dict = dict(zip(TARGET_LIST, range(len(TARGET_LIST))))


def process_file(img_file, label_file, output_folder):
    file_name = split(img_file)[-1].replace(".jpg", "")
    shutil.copy(img_file, join(output_folder, "images", file_name + ".jpg"))

    with open(label_file, "r") as f:
        data = json.load(f)
        target_mask = np.zeros((len(TARGET_LIST), data["imageHeight"], data["imageWidth"]))
        for index, shape in enumerate(data["shapes"]):
            target_img = Image.new("L", (data["imageWidth"], data["imageHeight"]), 0)
            target_index = target_dict[shape["label"]]
            if shape["label"] in TARGET_LIST:
                polygon = [(x, y) for x, y in shape["points"]]  # list to tuple
                ImageDraw.Draw(target_img).polygon(polygon, outline=1, fill=1)
            target_mask[target_index] = np.array(target_img)
    for i in range(0, len(TARGET_LIST)):
        cv2.imwrite(
            join(output_folder, "labels", f"{file_name}_{i}.png"),
            target_mask[i],
        )


def process_folder(img_folder, lable_folder, output_folder, num_processes=12):

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(join(output_folder, "images"), exist_ok=True)
    os.makedirs(join(output_folder, "labels"), exist_ok=True)
    img_files = os.listdir(img_folder)

    pool = multiprocessing.Pool(processes=num_processes)
    async_results = []

    for img_file in img_files:
        process_file(
            join(img_folder, img_file),
            join(lable_folder, img_file.replace(".jpg", ".json")),
            output_folder,
        )
        async_results.append(
            pool.starmap_async(
                process_file,
                (
                    (
                        join(img_folder, img_file),
                        join(lable_folder, img_file.replace(".jpg", ".json")),
                        output_folder,
                    ),
                ),
            )
        )

    _ = [a.get() for a in async_results]
    pool.close()
    pool.join()


if __name__ == "__main__":
    """
    Small Script to convert the  dataset into a suitable format
    Take all data from train and validation folders and save them in the same folder since cross
    validation is used instead of a single train-val split
    images are just copied, masks are converted from polygons in json files to a individual binary
    image for each class with {image_name}_{class_index}.png
    """

    root = "/media/l727r/data/Datasets/dacl20k/dacl10k_v2_devphase"
    output = "/media/l727r/data/Datasets/dacl20k/dacl10k_dataset"
    process_folder(join(root, "images", "train"), join(root, "annotations", "train"), output)
    process_folder(
        join(root, "images", "validation"), join(root, "annotations", "validation"), output
    )
