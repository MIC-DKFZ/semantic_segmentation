import os
import glob
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
import argparse

CLASSES = (
    "background",
    "aeroplane",
    "bag",
    "bed",
    "bedclothes",
    "bench",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bottle",
    "building",
    "bus",
    "cabinet",
    "car",
    "cat",
    "ceiling",
    "chair",
    "cloth",
    "computer",
    "cow",
    "cup",
    "curtain",
    "dog",
    "door",
    "fence",
    "floor",
    "flower",
    "food",
    "grass",
    "ground",
    "horse",
    "keyboard",
    "light",
    "motorbike",
    "mountain",
    "mouse",
    "person",
    "plate",
    "platform",
    "pottedplant",
    "road",
    "rock",
    "sheep",
    "shelves",
    "sidewalk",
    "sign",
    "sky",
    "snow",
    "sofa",
    "table",
    "track",
    "train",
    "tree",
    "truck",
    "tvmonitor",
    "wall",
    "water",
    "window",
    "wood",
)

mapping = np.array(
    [
        0,
        2,
        9,
        18,
        19,
        22,
        23,
        25,
        31,
        33,
        34,
        44,
        45,
        46,
        59,
        65,
        68,
        72,
        80,
        85,
        98,
        104,
        105,
        113,
        115,
        144,
        158,
        159,
        162,
        187,
        189,
        207,
        220,
        232,
        258,
        259,
        260,
        284,
        295,
        296,
        308,
        324,
        326,
        347,
        349,
        354,
        355,
        360,
        366,
        368,
        397,
        415,
        416,
        420,
        424,
        427,
        440,
        445,
        454,
        458,
    ]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()

    root_path = args.data_path

    outdir = os.path.join(root_path, "VOC2010_Context")
    if not os.path.exists(outdir):
        os.makedirs(os.path.join(outdir, "Annotations", "train"))
        os.makedirs(os.path.join(outdir, "Annotations", "val"))
        os.makedirs(os.path.join(outdir, "Images", "train"))
        os.makedirs(os.path.join(outdir, "Images", "val"))

    files = glob.glob(os.path.join(root_path, "trainval", "trainval", "*.mat"))

    with open(
        os.path.join(
            root_path,
            "VOCtrainval_03-May-2010",
            "VOCdevkit",
            "VOC2010",
            "ImageSets",
            "Main",
            "train.txt",
        )
    ) as file:
        train = file.readlines()
        train = [line.rstrip() for line in train]
    with open(
        os.path.join(
            root_path,
            "VOCtrainval_03-May-2010",
            "VOCdevkit",
            "VOC2010",
            "ImageSets",
            "Main",
            "val.txt",
        )
    ) as file:
        val = file.readlines()
        val = [line.rstrip() for line in val]

    print(len(train), len(val))
    print(len(files))

    print("## Covert Annoation Data ##")
    for file in tqdm(files):
        id = file.split("/")[-1].split(".")[0]

        label_file = os.path.join(root_path, "trainval", "trainval", id + ".mat")

        mat = scipy.io.loadmat(label_file)["LabelMap"]
        img = np.zeros(mat.shape)

        # values=np.unique(mat)
        for value in np.unique(mat):
            map = np.where(mapping == value)[0]
            if map.size > 0:
                img[mat == value] = map

        if id in train:
            outfile = os.path.join(outdir, "Annotations", "train", id + ".png")
            img_pil = Image.fromarray(np.uint8(img))
            img_pil.save(outfile)

        if id in val:
            outfile = os.path.join(outdir, "Annotations", "val", id + ".png")
            img_pil = Image.fromarray(np.uint8(img))
            img_pil.save(outfile)

    print("## Copy Image Data ##")
    for file in tqdm(files):
        id = file.split("/")[-1].split(".")[0]
        # print(os.path.join(root_path,file))

        img_file = os.path.join(
            root_path,
            "VOCtrainval_03-May-2010",
            "VOCdevkit",
            "VOC2010",
            "JPEGImages",
            id + ".jpg",
        )

        if id in train:
            outfile = os.path.join(outdir, "Images", "train", id + ".jpg")

            shutil.copy(img_file, outfile)

        if id in val:
            outfile = os.path.join(outdir, "Images", "val", id + ".jpg")

            shutil.copy(img_file, outfile)
