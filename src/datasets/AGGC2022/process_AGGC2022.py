import glob
import os
import random

import numpy as np
import dask.array as da
import zarr
import matplotlib.pyplot as plt
from numcodecs import Blosc, blosc
from openslide import OpenSlide, open_slide
import time
import cv2
import shutil
from skimage.transform import rescale
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 4.630.873.600


def load_and_resize_mask(slide: OpenSlide, target_shape: tuple = None):
    """
    Load, Resize and Save the Image as a png

    Parameters
    ----------
    slide: OpenSlide
    target_shape
        shape of the target for resize operation
    """
    slide_pil = slide.read_region([0, 0], 0, slide.level_dimensions[0]).convert("L")
    # slide_pil = slide.read_region([20000, 20000], 0, [10000, 8000]).convert("L")
    # image_pil = slide.read_region([0, 0], 0, [4000,4000]).convert('RGB')

    if slide.level_dimensions[0] != target_shape and target_shape is not None:
        slide_pil = slide_pil.resize(target_shape, resample=Image.NEAREST)

    return slide_pil


def load_and_resize_img(slide: OpenSlide, target_shape: tuple = None):
    """
    Load, Resize and Save the Image as a png

    Parameters
    ----------
    slide: OpenSlide
    target_shape
        shape of the target for resize operation
    """
    # print(slide.level_dimensions[0])
    # print(slide.dimensions)
    # print(slide.properties)
    slide_pil = slide.read_region([0, 0], 0, slide.level_dimensions[0]).convert("RGB")
    # slide_pil = slide.read_region([20000, 20000], 0, [10000, 8000]).convert("RGB")

    if slide.level_dimensions[0] != target_shape and target_shape is not None:
        slide_pil = slide_pil.resize(target_shape)

    return slide_pil


def save_np_as_zarr(image, output_file, chunksize=(512, 512, 3)):
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    image = zarr.array(image, chunks=chunksize, compressor=compressor)
    zarr.convenience.save(output_file, image)
    del image


def save_np_as_png(image_np, output_file):
    cv2.imwrite(output_file, image_np)


def save_pil_as_png(image_pil, output_file):
    image_pil.save(output_file)


if __name__ == "__main__":
    # Path to the Input files and the location where the output should be saved
    input_dir = "/home/l727r/Documents/E132-Projekte/Projects/2022_AGGC_challenge/GleasonGradMICCAIChallenge2022"
    output_dir = "/media/l727r/data/AGGC2022"

    # Define the Subset and the Classes
    subset = "Subset1"
    # global_classes=
    class_mapping = {"Stroma": 1, "Normal": 2, "G3": 3, "G4": 4, "G5": 5}
    all_classes = [
        "G5_Mask.tif",
        "G4_Mask.tif",
        "G3_Mask.tif",
        "Normal_Mask.tif",
        "Stroma_Mask.tif",
    ]

    output_dir = os.path.join(output_dir, subset)

    # List all images and masks in the directories
    image_dir = os.path.join(input_dir, subset + "_Train_image")
    mask_dir = os.path.join(input_dir, subset + "_Train_annotation", "Train")
    image_files = glob.glob(image_dir + "/*")
    mask_files = glob.glob(mask_dir + "/*")
    image_files.sort()
    mask_files.sort()

    print("{} Images and {} Masks are found".format(len(image_files), len(mask_files)))
    print("Start Processing")

    resize_img = True
    save_as = ".zarr"
    save_as = ".png"

    for img_file, mask_file in zip(image_files, mask_files):
        start_time = time.time()
        name = os.path.split(img_file)[-1].split(".tiff")[0]

        # This file is corrupt
        if name == "Subset1_Train_83" or name == "Subset1_Train_96":
            continue

        """
        Init the Openslide Object
        """
        slide_img = OpenSlide(img_file)  # open_slide(img_file)

        """
        Determine the Output shape, with a max length of max_length
        """
        if resize_img:
            width, height = slide_img.dimensions
            max_length = 4000
            if width > height:
                new_height = height * max_length / width
                new_width = max_length
            else:
                new_width = width * max_length / height
                new_height = max_length
            target_shape = (int(new_width), int(new_height))
        else:
            target_shape = slide_img.dimensions

        """
        Load, Resize and Save the Image
        """
        output_file_img = os.path.join(
            output_dir, "imgs_" + save_as.replace(".", ""), name + save_as
        )
        if not os.path.exists(output_file_img):
            print("{} : Loading Image with shape {}: ...".format(name, slide_img.dimensions))
            img_pil = load_and_resize_img(slide_img, target_shape)
            print("{} : Save Image with shape {} as {}: ...".format(name, img_pil.size, save_as))
            if save_as == ".png":
                save_pil_as_png(img_pil, output_file_img)
                del img_pil
            elif save_as == ".zarr":
                img_np = np.asarray(
                    img_pil, dtype=np.uint8
                )  # .astype(np.uint8)  # .transpose((1, 0, 2))
                del img_pil
                # img_np = img_np.transpose((1, 0, 2))
                save_np_as_zarr(img_np, output_file_img)
                del img_np

        slide_img.close()
        end_time = time.time()
        time_elapsed = int(end_time - start_time)
        print("{} : Finished with processing Image, took {} sec.".format(name, time_elapsed))

        """
        Load, Resize and Save the Mask
        """
        start_time = time.time()
        output_file_mask = os.path.join(
            output_dir, "masks_" + save_as.replace(".", ""), name + save_as
        )
        if not os.path.exists(output_file_mask):
            current_classes = os.listdir(mask_file)
            mask = np.zeros(target_shape, dtype=np.uint8).transpose()
            # mask = None
            # mask = np.zeros((10000, 8000), dtype=np.uint8).transpose()
            print(
                "{} : Create target mask of shape {}, which contains these classes: {}".format(
                    name, target_shape, current_classes
                )
            )
            for cl in all_classes:
                if cl in current_classes:
                    file_class = os.path.join(mask_file, cl)

                    slide_class = open_slide(file_class)
                    class_pil = load_and_resize_mask(slide_class, target_shape)
                    class_np = np.asarray(class_pil, dtype=np.uint8)
                    del class_pil
                    x, y = np.where(class_np == 255)
                    del class_np
                    cl_index = class_mapping[cl.replace("_Mask.tif", "")]
                    mask[x, y] = cl_index

            print("{} : Save Mask with shape {} as {}: ...".format(name, mask.shape, save_as))
            if save_as == ".png":
                save_np_as_png(mask, output_file_mask)
            elif save_as == ".zarr":
                save_np_as_zarr(mask, output_file_mask, chunksize=(512, 512))
            del mask
        end_time = time.time()
        time_elapsed = int(end_time - start_time)
        print("{} : Finished with processing Mask, took {} sec.".format(name, time_elapsed))
        # break
"""
# r_dim = np.array(slide_class.dimensions)

                    org_org = [0, 0]
                    size_org = np.array(slide_class.dimensions)

                    org_half_1 = [0, 0]
                    size_half_1 = [size_org[0], size_org[1] // 2]

                    org_half_2 = [0, size_org[1] // 2]
                    size_half_2 = [size_org[0], size_org[1] // 2]

                    # print("Original: origin {}, size {}".format(org_org, size_org))
                    # print("Half 1: origin {}, size {}".format(org_half_1, size_half_2))
                    # print("Half 2: origin {}, size {}".format(org_half_2, size_half_2))

                    class_pil = slide_class.read_region(
                        tuple(org_half_1), 0, tuple(size_half_1)
                    ).convert("L")
                    print(class_pil.size)
                    class_np_half_1 = np.asarray(class_pil, dtype=np.uint8)

                    class_pil = slide_class.read_region(
                        tuple(org_half_2), 0, tuple(size_half_2)
                    ).convert("L")
                    class_np_half_2 = np.asarray(class_pil, dtype=np.uint8)
                    print("Size", class_np_half_1.shape, class_np_half_2.shape)
                    class_np_half_1 = np.concatenate((class_np_half_1, class_np_half_2), axis=0)
                    print("Size concat", class_np_half_1.shape)

                    class_pil = slide_class.read_region([0, 0], 0, slide_class.dimensions).convert(
                        "L"
                    )
                    # class_np_org = np.asarray(class_pil, dtype=np.uint8)
                    # print("Size org", class_np_org.shape)

                    equal = class_np_half_1 == class_np_org
                    # print(np.all(equal))
                    print(np.array_equal(class_np_half_1, class_np_org))
                    quit()
"""
