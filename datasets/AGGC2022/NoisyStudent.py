import glob
import os

import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc, blosc

import tifffile
import torch.nn.functional as F
import torch
import cv2
import argparse


def save_np_as_png(image_np, output_file):
    cv2.imwrite(output_file, image_np)
    del image_np


def save_np_as_zarr(img_np, output_file, chunksize=(512, 512)):

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    image = zarr.array(img_np, chunks=chunksize, compressor=compressor)
    zarr.convenience.save(output_file, image)
    del image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Subset", default="Subset1")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--cluster", action="store_true")

    args = parser.parse_args()

    Subset = args.Subset
    if args.cluster:
        # "/dkfz/cluster/gpu/data/OE0441/l727r/AGGC2022"
        path_zarr = "/dkfz/cluster/gpu/data/OE0441/l727r/AGGC2022/" + Subset + "/masks_2/"
        path_tiff = "/dkfz/cluster/gpu/data/OE0441/l727r/aggc_preds/Noisy_" + Subset
        output_zarr = "/dkfz/cluster/gpu/data/OE0441/l727r/AGGC2022/" + Subset + "/masks_3/"
        output_png = "/dkfz/cluster/gpu/data/OE0441/l727r/AGGC2022/" + Subset + "/masks_png_3/"
    else:
        path_zarr = "/media/l727r/data/AGGC2022/" + Subset + "/masks_zarr/"
        path_tiff = "/media/l727r/data/AGGC2022/test"
        output_zarr = "/media/l727r/data/AGGC2022/" + Subset + "/masks_3/"
        output_png = "/media/l727r/data/AGGC2022/" + Subset + "/masks_png_3/"
    if not os.path.exists(output_zarr):
        os.makedirs(output_zarr)
    if not os.path.exists(output_png):
        os.makedirs(output_png)

    files_zarr = glob.glob(path_zarr + "*.zarr")
    if args.reverse:
        files_zarr.reverse()

    for file_zarr in tqdm(files_zarr):
        name = file_zarr.rsplit("/", 1)[1].replace(".zarr", "")
        file_tiff = os.path.join(path_tiff, name + ".tif")
        if not os.path.isfile(file_tiff):
            print("No tif for {}".format(file_zarr))
            continue
        if os.path.exists(output_zarr + name + ".zarr"):
            print("Already exits {}".format(file_zarr))
            continue
        print("Process {}".format(file_zarr))

        print("Open Tiff")
        img = tifffile.imread(file_tiff)
        img = img.astype(np.uint8)
        print("Open Zarr")
        gt = zarr.open(file_zarr, mode="r").astype(np.uint8)

        print("Process")
        # val = np.array(gt[:, :] == 0)
        # print(val.dtype)
        x, y = np.where(gt[:, :] != 0)
        # z = gt[x, y]
        img[x, y] = gt[x, y]

        img = img.astype(np.uint8)
        del gt

        save_np_as_zarr(img, output_zarr + name + ".zarr")
        print("Resize")
        width, height = img.shape
        max_length = 4000
        if width > height:
            new_height = height * max_length / width
            new_width = max_length
        else:
            new_width = width * max_length / height
            new_height = max_length
        target_shape = (int(new_width), int(new_height))

        img = (
            F.interpolate(
                torch.from_numpy(img).byte().unsqueeze(dim=0).unsqueeze(dim=0),
                size=target_shape,
            )
            .squeeze()
            .numpy()
        )

        save_np_as_png(img, output_png + name + ".png")
