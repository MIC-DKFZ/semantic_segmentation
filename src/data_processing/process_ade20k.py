import glob

from PIL import Image
import numpy as np
import argparse
import os
import pickle as pkl
import scipy
from tqdm import tqdm
import cv2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    root = args.data_path
    splits = ["training", "validation"]
    "annotations/"
    os.makedirs(os.path.join(root, "annotations_150classes"), exist_ok=True)
    os.makedirs(os.path.join(root, "annotations_150classes", "training"), exist_ok=True)
    os.makedirs(os.path.join(root, "annotations_150classes", "validation"), exist_ok=True)
    for split in splits:
        files = glob.glob(os.path.join(root, "annotations", split, "*"))
        for file in tqdm(files):
            mask = cv2.imread(file, -1) - 1
            cv2.imwrite(file.replace("annotations", "annotations_150classes"), mask)
