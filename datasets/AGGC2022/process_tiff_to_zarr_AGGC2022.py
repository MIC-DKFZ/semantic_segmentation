import glob
import os

import numpy as np
import zarr
from numcodecs import Blosc, blosc
from openslide import OpenSlide, open_slide
import time
from PIL import Image
from multiprocessing import Pool
import logging
import argparse

logging.getLogger(__name__).addHandler(logging.StreamHandler())
Image.MAX_IMAGE_PIXELS = None  # 4.630.873.600


def save_np_as_zarr(image, output_file, chunksize=(512, 512, 3)):
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    image = zarr.array(image, chunks=chunksize, compressor=compressor)
    zarr.convenience.save(output_file, image)
    del image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="all")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    args = parser.parse_args()

    # Define the Subset and the Classes
    if args.subset == "all":
        subsets = ["Subset1", "Subset2", "Subset3"]
    elif args.subset == "S1":
        subsets = ["Subset1"]
    elif args.subset == "S2":
        subsets = ["Subset2"]
    elif args.subset == "S3":
        subsets = ["Subset3"]
    start_id = args.start_id
    end_id = args.end_id
    # quit()
    cluster_input_dir = "/dkfz/cluster/gpu/data/OE0441/l727r/GleasonGradMICCAIChallenge2022"
    cluster_output_dir = "/dkfz/cluster/gpu/data/OE0441/l727r/AGGC2022"

    local_input_dir = "/home/l727r/Documents/E132-Projekte/Projects/2022_AGGC_challenge/GleasonGradMICCAIChallenge2022"
    local_output_dir = "/media/l727r/data/AGGC2022"

    if os.path.isdir(local_input_dir) and os.path.isdir(local_output_dir):
        logging.info("Local Environment")
        print("Local Environment")
        input_dir = local_input_dir
        output_directory = local_output_dir
    elif os.path.isdir(cluster_input_dir) and os.path.isdir(cluster_output_dir):
        logging.info("Cluster Environment")
        logging.info("Cluster Environment")
        input_dir = cluster_input_dir
        output_directory = cluster_output_dir
    else:
        logging.info("Local and Cluster Path do not exist!!!")
        print("Local and Cluster Path do not exist!!!")
        quit()
    # Path to the Input files and the location where the output should be saved

    # input_dir = "/home/l727r/Documents/E132-Projekte/Projects/2022_AGGC_challenge/GleasonGradMICCAIChallenge2022"
    # output_directory = "/media/l727r/data/AGGC2022"

    # global_classes=
    class_mapping = {"Stroma": 1, "Normal": 2, "G3": 3, "G4": 4, "G5": 5}
    all_classes = [
        "G5_Mask.tif",
        "G4_Mask.tif",
        "G3_Mask.tif",
        "Normal_Mask.tif",
        "Stroma_Mask.tif",
    ]
    all_classes.reverse()
    for subset in subsets:
        # Create Folders if they not exist
        output_dir = os.path.join(output_directory, subset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, "imgs")):
            os.makedirs(os.path.join(output_dir, "imgs"))
        if not os.path.exists(os.path.join(output_dir, "masks_2")):
            os.makedirs(os.path.join(output_dir, "masks_2"))
        output_dir = os.path.join(output_dir)

        # List all images and masks in the directories
        image_dir = os.path.join(input_dir, subset + "_Train_image")
        mask_dir = os.path.join(input_dir, subset + "_Train_annotation", "Train")
        if subset == "Subset3":
            image_files = glob.glob(image_dir + "/*/*")
            mask_files = glob.glob(mask_dir + "/*/*")
        else:
            image_files = glob.glob(image_dir + "/*")
            mask_files = glob.glob(mask_dir + "/*")
        mask_names = [
            mask.rsplit("/", 1)[-1].replace(".tif", "").replace(".tiff", "") for mask in mask_files
        ]
        image_files = [
            img for img in image_files if img.rsplit("/")[-1].replace(".tiff", "") in mask_names
        ]
        image_files.sort()
        mask_files.sort()
        if end_id == -1:
            image_files = image_files[start_id:]
            mask_files = mask_files[start_id:]
        else:
            image_files = image_files[start_id:end_id]
            mask_files = mask_files[start_id:end_id]

        # image_files = [
        #    "/home/l727r/Documents/E132-Projekte/Projects/2022_AGGC_challenge/GleasonGradMICCAIChallenge2022/Subset3_Train_image/Philips/Subset3_Train_12_Philips.tiff"
        # ]
        # mask_files = [
        #    "/home/l727r/Documents/E132-Projekte/Projects/2022_AGGC_challenge/GleasonGradMICCAIChallenge2022/Subset3_Train_annotation/Train/Philips/Subset3_Train_12_Philips/"
        # ]
        print(
            "For {}: {} Images and {} Masks are found".format(
                subset, len(image_files), len(mask_files)
            )
        )

        print("Start Processing")
        # continue
        for img_file, mask_file in zip(image_files, mask_files):
            print(img_file, mask_file)
            name = os.path.split(img_file)[-1].split(".tiff")[0]
            # if "Philips" in name:
            #    continue

            output_file_img = os.path.join(output_dir, "imgs", name + ".zarr")
            output_file_mask = os.path.join(output_dir, "masks_2", name + ".zarr")
            print(output_file_img)
            # This file is corrupt
            if name == "Subset1_Train_83":
                continue
            # elif name != "Subset1_Train_96":
            #    continue
            """
            Init the Openslide Object
            """
            slide_img = OpenSlide(img_file)  # open_slide(img_file)

            """
            Determine the Output shape, with a max length of max_length
            """
            target_shape = slide_img.dimensions
            if name == "Subset1_Train_96":
                target_shape = (target_shape[0] - 1500, target_shape[1])

            """
            Image: Load, Resize and Save
            """
            start_time = time.time()
            if not os.path.exists(output_file_img):  # and subset != "Subset3":
                print("{} : Loading Image with shape {}: ...".format(name, slide_img.dimensions))
                # Reading Image
                if name == "Subset1_Train_96":
                    w, h = slide_img.level_dimensions[0]
                    img_pil = slide_img.read_region([1500, 0], 0, (w - 1500, h - 0)).convert("RGB")
                else:
                    img_pil = slide_img.read_region(
                        [0, 0], 0, slide_img.level_dimensions[0]
                    ).convert("RGB")
                print("{} : Save Image with shape {}: ...".format(name, img_pil.size))
                # Convert to numpy
                img_np = np.asarray(img_pil, dtype=np.uint8)

                del img_pil
                # Save img as .zarr
                save_np_as_zarr(img_np, output_file_img)
                del img_np

            slide_img.close()
            end_time = time.time()
            time_elapsed = int(end_time - start_time)
            print("{} : Finished with processing Image, took {} sec.".format(name, time_elapsed))

            """
            Mask: Load, Resize and Save
            """
            start_time = time.time()

            if not os.path.exists(output_file_mask):  # and subset != "Subset3":
                current_classes = os.listdir(mask_file)
                mask = np.zeros(target_shape, dtype=np.uint8).transpose()
                print(
                    "{} : Create target mask of shape {}, which contains these classes: {}".format(
                        name, target_shape, current_classes
                    )
                )
                for cl in all_classes:
                    if cl in current_classes:
                        file_class = os.path.join(mask_file, cl)

                        slide_class = open_slide(file_class)
                        # class_pil = load_and_resize_mask(slide_class, target_shape)
                        if name == "Subset1_Train_96":
                            w, h = slide_class.level_dimensions[0]
                            class_pil = slide_class.read_region(
                                [1500, 0], 0, (w - 1500, h - 0)
                            ).convert("L")
                        else:
                            class_pil = slide_class.read_region(
                                [0, 0], 0, slide_class.level_dimensions[0]
                            ).convert("L")
                        class_np = np.asarray(class_pil, dtype=np.uint8)
                        del class_pil
                        x, y = np.where(class_np == 255)
                        del class_np
                        cl_index = class_mapping[cl.replace("_Mask.tif", "")]
                        mask[x, y] = cl_index

                print("{} : Save Mask with shape {}: ...".format(name, mask.shape))
                # Save as zarr
                save_np_as_zarr(mask, output_file_mask, chunksize=(512, 512))
                del mask
            end_time = time.time()
            time_elapsed = int(end_time - start_time)
            print("{} : Finished with processing Mask, took {} sec.".format(name, time_elapsed))
            # break
