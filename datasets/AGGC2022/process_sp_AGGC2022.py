import glob
import os
import random
import json
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
from skimage import measure
from skimage import filters
import pickle as pkl

Image.MAX_IMAGE_PIXELS = None  # 4.630.873.600

if __name__ == "__main__":
    path_imgs = "/home/l727r/Documents/E132-Projekte/Projects/2022_AGGC_challenge/GleasonGradMICCAIChallenge2022"
    path_masks = "/media/l727r/data/AGGC2022/Subset3/masks_png"
    output_directory = "/media/l727r/data/AGGC2022/"
    # output_directory = "/media/l727r/data/AGGC2022/Subset1/boxes"
    path_masks_org = path_masks
    subsets = ["Subset1", "Subset2", "Subset3"]
    subsets = ["Subset3"]
    for subset in subsets:
        # Create Folders if they not exist
        path_bb = os.path.join(output_directory, subset)
        if not os.path.exists(path_bb):
            os.makedirs(path_bb)
        if not os.path.exists(os.path.join(path_bb, "sample_points")):
            os.makedirs(os.path.join(path_bb, "sample_points"))
        path_sp = os.path.join(path_bb, "sample_points")

        image_dir = os.path.join(path_imgs, subset + "_Train_image")
        path_masks = os.path.join("/media/l727r/data/AGGC2022", subset, "masks_png", "*.png")
        # mask_dir = os.path.join(input_dir, subset + "_Train_annotation", "Train")
        if subset == "Subset3":
            imgs = glob.glob(image_dir + "/*/*")
        else:
            imgs = glob.glob(image_dir + "/*")
        masks = glob.glob(path_masks)

        mask_names = [mask.rsplit("/", 1)[-1].replace(".png", "") for mask in masks]
        # img_names = [img for img in imgs ]
        # print(mask_names)
        # print(img_names)
        imgs = [img for img in imgs if img.rsplit("/")[-1].replace(".tiff", "") in mask_names]

        imgs.sort()
        masks.sort()

        # print(imgs)
        # print(masks)

        print("For {}: {} Images and {} Masks are found".format(subset, len(imgs), len(masks)))
        # continue
        for mask_path, img_path in zip(masks, imgs):
            mask_path = os.path.join(
                path_masks_org, img_path.rsplit("/", 1)[-1].replace("tiff", "png")
            )
            print(mask_path, img_path)
            name = os.path.split(mask_path)[-1].split(".png")[0]
            print(name)
            # q index = 3
            # index = 5
            # print(imgs[index])
            # img = cv2.imread(imgs[index])
            slide_img = OpenSlide(img_path)
            width_org, height_org = slide_img.level_dimensions[0]
            # print(slide_img.level_dimensions[0])
            slide_img.close()

            mask = cv2.imread(mask_path, -1)
            # print(mask.shape)
            width, height = mask.shape
            # print(width, height, width_org, height_org)

            scale_factor1 = width_org / height
            scale_factor2 = height_org / width
            scale_factor = (scale_factor1 + scale_factor2) / 2
            # print(scale_factor)
            # quit()
            sp = {}
            for un_mask in np.unique(mask):
                if un_mask == 0:
                    continue
                x, y = np.where(mask == un_mask)
                number_points = 10000
                if len(x) > number_points:
                    idx = np.random.choice(np.arange(len(x)), number_points, replace=False)
                    x = x[idx]
                    y = y[idx]
                x = np.array(x * scale_factor, dtype=int)
                y = np.array(y * scale_factor, dtype=int)
                # p = np.random.choice(zip(x, y), 10000)
                print(un_mask, len(x), len(y))
                sp[str(un_mask)] = np.array([x, y])

            sp = {k: v.tolist() for k, v in sp.items()}
            # name = mask_path.rsplit("/")[-1].replace(".png", ".json")
            name = mask_path.rsplit("/")[-1].replace(".png", ".pkl")
            with open(os.path.join(path_sp, name), "wb") as fp:
                # json.dump(sp, fp)
                pkl.dump(sp, fp)
                """if un_mask == 0:
                    continue
                mask_binary = np.zeros(mask.shape)
                mask_binary[mask == un_mask] = 1
                all_labels = measure.label(mask_binary, background=0)
                # print(all_labels.shape, np.unique(all_labels))
                bbs_class = []
                for un_labels in np.unique(all_labels):
                    if un_labels != 0:
                        x, y = np.where(all_labels == un_labels)
                        # print(min(x), min(y), max(x), max(y))
                        x = (x * scale_factor).astype(int)
                        y = (y * scale_factor).astype(int)
                        # print(min(x), min(y), max(x), max(y))
                        # bbs_class.append((min(x), min(y), max(x), max(y)))
                        bbs_class.append(
                            (int(min(x)), int(min(y)), int(np.ceil(max(x))), int(np.ceil(max(y))))
                        )
                        # print(bbs_class)
                        # print(np.array(bbs_class))
                        # cv2.rectangle(bb_map, (min(x), min(y)), (max(x), max(y)), 1, -1)
                bb[str(un_mask)] = np.array(bbs_class, dtype=int)
                # bb[str(un_mask)] = bbs_class
            name = mask_path.rsplit("/")[-1].replace(".png", ".json")
            print(name, scale_factor)
            bb = {k: v.tolist() for k, v in bb.items()}
            with open(os.path.join(path_bb, name), "w") as fp:
                json.dump(bb, fp)
            """
            # quit()
    quit()
    imgs = glob.glob(os.path.join(path_imgs, "*.tiff"))
    masks = glob.glob(os.path.join(path_masks, "*.png"))
    imgs.sort()
    masks.sort()
    print(len(imgs), len(masks))
    for mask_path, img_path in zip(masks, imgs):
        # q index = 3
        # index = 5
        # print(imgs[index])
        # img = cv2.imread(imgs[index])
        slide_img = OpenSlide(img_path)
        width_org, height_org = slide_img.level_dimensions[0]
        print(slide_img.level_dimensions[0])
        slide_img.close()

        mask = cv2.imread(mask_path, -1)
        print(mask.shape)
        width, height = mask.shape
        print(width, height, width_org, height_org)

        scale_factor1 = width_org / height
        scale_factor2 = height_org / width
        scale_factor = (scale_factor1 + scale_factor2) / 2
        print(scale_factor, scale_factor1, scale_factor2)
        # quit()
        bb = {}
        for un_mask in np.unique(mask):
            if un_mask == 0:
                continue
            mask_binary = np.zeros(mask.shape)
            mask_binary[mask == un_mask] = 1
            all_labels = measure.label(mask_binary, background=0)
            # print(all_labels.shape, np.unique(all_labels))
            bbs_class = []
            for un_labels in np.unique(all_labels):
                if un_labels != 0:
                    x, y = np.where(all_labels == un_labels)
                    print(min(x), min(y), max(x), max(y))
                    x = (x * scale_factor).astype(int)
                    y = (y * scale_factor).astype(int)
                    print(min(x), min(y), max(x), max(y))
                    bbs_class.append((min(x), min(y), max(x), max(y)))
                    # cv2.rectangle(bb_map, (min(x), min(y)), (max(x), max(y)), 1, -1)
            bb[str(un_mask)] = np.array(bbs_class)
        name = mask_path.rsplit("/")[-1].replace(".png", ".pkl")
        print(name)
        bb = {k: v.tolist() for k, v in bb.items()}
        with open(os.path.join(path_bb, name), "w") as fp:
            # json.dump(bb, fp)
            pkl.dump(bb, fp)
        # break
    quit()
    # n = 12
    # l = 256
    # mask = filters.gaussian(mask, sigma=l / (4.0 * n))
    # all_labels = measure.label(mask, background=0)
    # print(all_labels.shape, np.unique(all_labels))
    # bb = []
    mask = mask * 50
    color_mapping = {
        "1": [255, 0, 0],
        "2": [0, 255, 0],
        "3": [0, 0, 255],
        "4": [255, 255, 0],
        "5": [255, 0, 255],
    }
    print(bb)
    for key, value in bb.items():
        print(key, value)
        color = color_mapping[str(key)]
        for val in value:
            x_min, y_min, x_max, y_max = val

            cv2.rectangle(mask, (y_min, x_min), (y_max, x_max), color, 3)

    cv2.namedWindow("Window_mask", cv2.WINDOW_NORMAL)

    # cv2.namedWindow("Window", cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("Window", img)
    cv2.imshow("Window_mask", mask)
    # cv2.imshow("Window_bb",)
    cv2.waitKey()
