import glob
import os

import numpy as np
from openslide import OpenSlide, open_slide
import time
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 4.630.873.600


def save_np_as_png(image_np, output_file):
    cv2.imwrite(output_file, image_np)


def save_pil_as_png(image_pil, output_file):
    image_pil.save(output_file)


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
    # w, h = slide.level_dimensions[0]
    # slide_pil = slide.read_region([1500, 0], 0, (w - 1500, h - 0)).convert("RGB")
    # slide_pil = slide.read_region([20000, 20000], 0, [10000, 8000]).convert("RGB")

    if slide.level_dimensions[0] != target_shape and target_shape is not None:
        slide_pil = slide_pil.resize(target_shape)

    return slide_pil


if __name__ == "__main__":
    # from multiprocessing import Pool

    # p = Pool(16)
    # Path to the Input files and the location where the output should be saved
    input_dir = "/home/l727r/Documents/E132-Projekte/Projects/2022_AGGC_challenge/GleasonGradMICCAIChallenge2022"
    output_directory = "/media/l727r/data/AGGC2022"

    # Define the Subset and the Classes
    subsets = ["Subset1", "Subset2", "Subset3"]

    # global_classes=
    class_mapping = {"Stroma": 1, "Normal": 2, "G3": 3, "G4": 4, "G5": 5}
    all_classes = [
        "G5_Mask.tif",
        "G4_Mask.tif",
        "G3_Mask.tif",
        "Normal_Mask.tif",
        "Stroma_Mask.tif",
    ]

    for subset in subsets:
        # Create Folders if they not exist
        output_dir = os.path.join(output_directory, subset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, "imgs_png")):
            os.makedirs(os.path.join(output_dir, "imgs_png"))
        if not os.path.exists(os.path.join(output_dir, "masks_png")):
            os.makedirs(os.path.join(output_dir, "masks_png"))
        # output_dir = os.path.join(output_dir)

        # List all images and masks in the directories
        image_dir = os.path.join(input_dir, subset + "_Train_image")
        mask_dir = os.path.join(input_dir, subset + "_Train_annotation", "Train")
        if subset == "Subset3":
            image_files = glob.glob(image_dir + "/*/*")
            mask_files = glob.glob(mask_dir + "/*/*")
        else:
            image_files = glob.glob(image_dir + "/*")
            mask_files = glob.glob(mask_dir + "/*")

        mask_names = [mask.rsplit("/", 1)[-1].replace(".png", "") for mask in mask_files]
        # img_names = [img for img in imgs ]
        # print(mask_names)
        # print(img_names)
        image_files = [
            img for img in image_files if img.rsplit("/")[-1].replace(".tiff", "") in mask_names
        ]
        image_files.sort()
        mask_files.sort()

        print(
            "For {}: {} Images and {} Masks are found".format(
                subset, len(image_files), len(mask_files)
            )
        )

        print("Start Processing")
        resize_img = True
        for img_file, mask_file in zip(image_files, mask_files):

            name = os.path.split(img_file)[-1].split(".tiff")[0]
            output_file_img = os.path.join(output_dir, "imgs_png", name + ".png")
            output_file_mask = os.path.join(output_dir, "masks_png", name + ".png")
            # This file is corrupt
            if (
                name == "Subset1_Train_83"  # corrupy
                # or "Philips" in name
                or name == "Subset1_Train_96"  # corrupt shape?
                # or name == "Subset3_Train_25_Zeiss"  # to large
            ):
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
            Image: Load, Resize and Save
            """
            start_time = time.time()
            if not os.path.exists(output_file_img):
                print("{} : Loading Image with shape {}: ...".format(name, slide_img.dimensions))
                # Reading Image
                img_pil = load_and_resize_img(slide_img, target_shape)

                print("{} : Save Image with shape {}: ...".format(name, img_pil.size))
                save_pil_as_png(img_pil, output_file_img)
                del img_pil

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
                        class_pil = load_and_resize_mask(slide_class, target_shape)
                        class_np = np.asarray(class_pil, dtype=np.uint8)
                        del class_pil
                        x, y = np.where(class_np == 255)
                        del class_np
                        cl_index = class_mapping[cl.replace("_Mask.tif", "")]
                        mask[x, y] = cl_index

                print("{} : Save Mask with shape {}: ...".format(name, mask.shape))
                # Save as zarr
                save_np_as_png(mask, output_file_mask)
                del mask
            end_time = time.time()
            time_elapsed = int(end_time - start_time)
            print("{} : Finished with processing Mask, took {} sec.".format(name, time_elapsed))
    # p.close()
    # p.join()

    # break
    # continue
