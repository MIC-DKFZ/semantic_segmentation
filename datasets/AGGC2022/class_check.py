import glob
import os
import random
import json
import numpy as np
from tqdm import tqdm
import pickle as pkl
import dask.array as da
import zarr
import matplotlib.pyplot as plt
from numcodecs import Blosc, blosc
from openslide import OpenSlide, open_slide
import time
import tifffile

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2
import shutil
from skimage.transform import rescale
from PIL import Image
from skimage import measure
from skimage import filters

Image.MAX_IMAGE_PIXELS = None


def save_pil_as_png(image_pil, output_file):
    print("Save Images")
    image_pil.save(output_file)


def load_and_resize_mask(slide: OpenSlide, target_shape: tuple = None, name=None):
    """
    Load, Resize and Save the Image as a png

    Parameters
    ----------
    slide: OpenSlide
    target_shape
        shape of the target for resize operation
    """
    print("Load Image")
    if name == "Subset1_Train_96":
        w, h = slide.level_dimensions[0]
        slide_pil = slide.read_region([1500, 0], 0, (w - 1500, h - 0)).convert("L")
    else:
        slide_pil = slide.read_region([0, 0], 0, slide.level_dimensions[0]).convert("L")
    # slide_pil = slide.read_region([20000, 20000], 0, [10000, 8000]).convert("L")
    # image_pil = slide.read_region([0, 0], 0, [4000,4000]).convert('RGB')
    print("Resize Image")
    if slide.level_dimensions[0] != target_shape and target_shape is not None:
        slide_pil = slide_pil.resize(target_shape, resample=Image.NEAREST)

    return slide_pil


if __name__ == "__main__":
    """dir = "/home/l727r/Documents/cluster-checkpoints/aggc_preds/Noisy_Subset3/*.tif"
    files = glob.glob(dir)
    for file in files:
        print(file)
        try:
            # img = tifffile.imread(file)
            # print(img.shape)
            # print("X")
            slide_class = open_slide(file)
            width, height = slide_class.dimensions
            print(width, height)
        except Exception as e:
            print(e)
        # break
    quit()
    input_dir = "/dkfz/cluster/gpu/data/OE0441/l727r/AGGC2022/Subset1/mask_noisy"
    input_dir = "/home/l727r/Documents/cluster-data/AGGC2022/Subset1/mask_noisy"
    output_dir = "/dkfz/cluster/gpu/data/OE0441/l727r/AGGC2022/mask_noisy_png"
    output_dir = "/media/l727r/data/AGGC2022/Subset1/mask_noisy_png/test.tif"
    files = glob.glob(input_dir + "/*")
    print(len(files))
    for file in files:
        print(file)
        name = file.split("/")[-1].replace(".tif", ".png")
        output_file_mask = os.path.join(output_dir, name)
        print(output_file_mask)
        slide_class = open_slide(file)
        width, height = slide_class.dimensions

        max_length = 4000
        if width > height:
            new_height = height * max_length / width
            new_width = max_length
        else:
            new_width = width * max_length / height
            new_height = max_length
        target_shape = (int(new_width), int(new_height))
        print("{} original size, {} target size".format(slide_class.dimensions, target_shape))
        class_pil = load_and_resize_mask(slide_class, target_shape)
        save_pil_as_png(class_pil, output_file_mask)

    quit()
    """
    """
    
    root_masks = "/media/l727r/data/AGGC2022/"
    root_masks = "/media/l727r/data/AGGC2022/"
    #root_masks_2 = "/home/l727r/Documents/cluster-data/AGGC2022/"
    files = glob.glob(root_masks + "Subset*/masks_png_2/*.png")
    files_2 = glob.glob(root_masks + "Subset*/masks_png_3/*.png")
    Yes = 0
    No = 0
    Pix = 0
    print(len(files), len(files_2))
    for file in files:
        name = file.rsplit("/", 1)[1]
        subset = name.split("_")[0]

        file_2 = root_masks + subset + "/masks_png/" + name
        if file_2 not in files_2:
            continue
        img = cv2.imread(file)
        img_2 = cv2.imread(file_2)
        status = np.all(img == img_2)
        if status:
            Yes += 1
            equ = 0
        else:
            No += 1
            x, y = np.unique(img == img_2, return_counts=True)
            # print(x)
            equ = y[0]
            Pix += equ
        print(name, status, equ, Yes, No)
    print("Equal", Yes)
    print("Not Equal", No)
    print("Pixel", Pix / No)
    print(name, subset)
    quit()
    """
    """path = "/home/l727r/Documents/E132-Projekte/Projects/2022_AGGC_challenge/GleasonGradMICCAIChallenge2022"
    class_list = []
    number_points = 10000
    for c in ["Stroma", "Normal", "G3", "G4", "G5"]:
        list = []
        files_1 = glob.glob(os.path.join(path, "*", "Train", "*", c + "_Mask.tif"))
        files_2 = glob.glob(os.path.join(path, "*", "Train", "*", "*", c + "_Mask.tif"))
        files = files_1 + files_2
        files.sort()
        print(c, len(files))
        for file in tqdm(files):
            name = file.split("/")[-2]
            subset = name.split("_")[0]
            print(name, subset)
            subset = name.split("_")[0]
            outfile_pts = os.path.join(
                "/media/l727r/data/AGGC2022", subset, "sample_points_new", name + ".pkl"
            )
            print(name, subset, outfile_pts)

            mask = cv2.imread(file, -1)
            x, y = np.where(mask != 0)

            idx = np.random.choice(np.arange(len(x)), number_points, replace=False)
            x = x[idx]
            y = y[idx]

            sp = {k: v.tolist() for k, v in sp.items()}
            with open(outfile_pts, "wb") as fp:
                pkl.dump(sp, fp)

            # print(np.unique(mask), mask.shape)
            # list.append((name, int(len(x)), np.sum(mask / 255)))
            # print(list)
        # class_list.append(list)
        # break
    quit()
    """
    # Stroma 272
    # Normal 265
    # G3 260
    # G4 245
    # G5 40
    # New
    # Stroma 272
    # Normal 264
    # G3 251
    # G4 254
    # G5 47
    # NoisyStu
    # Stroma 285
    # Normal 285
    # G3 283
    # G4 280
    # G5 223
    # S1 104,104,104,104,97
    # S1 37,37,35,32,4
    # S1 144,144,144,144,122
    root_masks = "/media/l727r/data/AGGC2022/"
    output_directory = "/media/l727r/data/AGGC2022/"
    subsets = ["Subset1", "Subset2", "Subset3"]
    subsets = ["Subset2"]
    # subsets = ["Subset1"]
    # subsets = ["Subset3"]
    class_files = [[], [], [], [], []]
    prefix = {
        "Subset1": "/Subset1/imgs/",
        "Subset2": "/Subset2/imgs/",
        "Subset3": "/Subset3/imgs/",
    }
    for subset in subsets:
        path_masks = os.path.join(root_masks, subset, "masks_png_3", "*.png")
        # mask_dir = os.path.join(input_dir, subset + "_Train_annotation", "Train")
        masks_files = glob.glob(path_masks)
        masks_files.sort()
        print("For {}: {} Masks are found".format(subset, len(masks_files)))

        for mask_file in tqdm(masks_files):
            name = mask_file.rsplit("/", 1)[1]
            subset = name.split("_")[0]
            name = prefix[subset] + name
            mask = cv2.imread(mask_file, -1)
            classes, count = np.unique(mask, return_counts=True)
            # print(classes)
            for cl, cou in zip(classes, count):
                if cl != 0:
                    class_files[cl - 1].append((name, int(cou)))
            # print(class_files)
            # print(name, mask.shape)

    class_mapping = ["Stroma", "Normal", "G3", "G4", "G5"]
    # sp = {k: v.tolist() for k, v in sp.items()}
    # print(class_files)
    output = {}
    for name, classes in zip(class_mapping, class_files):
        print(name, len(classes))
        output[name] = classes
    # print(output)
    # output = {k: v.tolist() for k, v in output.items()}
    # name = mask_path.rsplit("/")[-1].replace(".png", ".json")
    # name = mask_path.rsplit("/")[-1].replace(".png", ".pkl")
    quit()
    with open(
        os.path.join(output_directory, "Images_per_Class_3.json"), "w", encoding="utf-8"
    ) as fp:
        json.dump(output, fp)
    quit()
    # break

    # print(imgs)
    # print(masks)
    # for mask_path in masks:
    #    mask_path = os.path.join(
    #        path_masks_org, mask_path.rsplit("/", 1)[-1].replace("tiff", "png")
    #    )
    # mask = cv2.imread(mask_path, -1)
    # classes = np.unique(mask)
    # print(classes)
