import numpy as np
from numpy import ndarray
import random
import albumentations.augmentations.geometric.functional as aF
import cv2


def split_by_ID(files: list, ids: list = None, num_folds: int = 5) -> list:
    """
    Split the ids into num_folds different folds. Afterward assign each file in files to the split
    which contains the id which belongs to the file. Outputformat [{"train":[...],"val:[...]}, ...].
    """
    if ids is None:
        ids = files
    splits = []
    for i in range(0, num_folds):
        val_ids = ids[i::num_folds]
        train_ids = [ID for ID in ids if ID not in val_ids]

        val_samples = [file for file in files if any(val_id in file for val_id in val_ids)]
        train_samples = [file for file in files if any(train_id in file for train_id in train_ids)]

        # Sanity Check that no img is in both sets
        for vs in val_samples:
            if vs in train_samples:
                print(f"WARING: There is a sample in train AND val set for fold {i}: {vs}")

        print(
            f"Split {i}, #Train: {len(train_samples)}, #Val: {len(val_samples)}, #total"
            f" {len(train_samples)+len(val_samples)}"
        )
        splits.append({"train": train_samples, "val": val_samples})
    return splits


def pad_if_needed(img: ndarray, mask: ndarray, patch_size: tuple, border_mode: int = 4) -> tuple:
    """
    Pad img and mask to have at least the size of patch_size. By default, border moder 4
    (cv2.BORDER_REFLECT_101) is used.
    """

    def get_pad_params(dim_img, dim_patch):
        if dim_img < dim_patch:
            pad_up = int((dim_patch - dim_img) / 2)
            pad_low = dim_patch - dim_img - pad_up
        else:
            pad_up = 0
            pad_low = 0
        return pad_up, pad_low

    w, h = img.shape[:2]
    pad_top, pad_bottom = get_pad_params(w, patch_size[0])
    pad_left, pad_right = get_pad_params(h, patch_size[1])

    img = aF.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right, border_mode)
    mask = aF.pad_with_params(mask, pad_top, pad_bottom, pad_left, pad_right, border_mode)

    return img, mask


def scale(img: ndarray, mask: ndarray, scale_factor: float) -> tuple:
    """
    Scale img and mask by the scale_factor
    """
    if scale_factor != 1:
        img = aF.scale(img, scale_factor, cv2.INTER_LINEAR)
        mask = aF.scale(mask, scale_factor, cv2.INTER_NEAREST)
    return img, mask


def keypoint_scale(img: ndarray, mask: ndarray, scale_factor: float, keypoint: tuple) -> tuple:
    """
    Scale img, mask and keypoint by the scale_factor
    """
    if scale_factor != 1:
        img, mask = scale(img, mask, scale_factor)
        keypoint = (int(keypoint[0]) * scale_factor, int(keypoint[1] * scale_factor))
    return img, mask, keypoint


def random_crop(img: ndarray, mask: ndarray, patch_size: tuple) -> tuple:
    """
    Randomly Crop the img and mask to patch_size. Since the shape of the output is
    min(img_size,patch_size) the output can be smaller than the patch_size.
    """
    w, h, _ = img.shape

    # Get a random point
    x_min = np.random.randint(0, max(1, w - patch_size[0]))
    y_min = np.random.randint(0, max(1, h - patch_size[1]))
    # Get the max value
    x_max = int(min(x_min + patch_size[0], w))
    y_max = int(min(y_min + patch_size[1], h))

    img_cropped = img[x_min:x_max, y_min:y_max]
    mask_cropped = mask[x_min:x_max, y_min:y_max]

    return img_cropped, mask_cropped


def keypoint_crop(img: ndarray, mask: ndarray, patch_size: tuple, keypoint: tuple) -> tuple:
    """
    Crop the img and mask to patch_size around the keypoint. Since the shape of the output is
    min(img_size,patch_size) the output can be smaller than the patch_size.
    """
    w, h, _ = img.shape

    # Clip the patch to be inside the image with min=0 and max=(w,h) - patch_size
    x_min = int(max(min(keypoint[0] - np.ceil(patch_size[0] / 2), w - patch_size[0]), 0))
    y_min = int(max(min(keypoint[1] - np.ceil(patch_size[1] / 2), h - patch_size[1]), 0))

    # Compute
    x_max = int(min(x_min + patch_size[0], w))
    y_max = int(min(y_min + patch_size[1], h))

    # Copping the image
    img_cropped = img[x_min:x_max, y_min:y_max]
    mask_cropped = mask[x_min:x_max, y_min:y_max]

    return img_cropped, mask_cropped


def random_scale_crop(
    img: ndarray, mask: ndarray, patch_size: tuple, scale_limit: tuple = (0, 0)
) -> tuple:
    """
    Scale img and mask by a random scale factor in the range of scale_limit. Then randomly crop the
    img and mask to patch_size around the keypoint and pad if needed to have the desired patch_size.
    """
    # Conversion is needed to be consistent to albumentations notation
    scale_limit = (scale_limit[0] + 1, scale_limit[1] + 1)
    # Randomly scale img and mask
    scale_factor = random.uniform(*scale_limit)
    img, mask = scale(img, mask, scale_factor)
    # Randomly crop img and mask and pad_if_needed afterward to ensure img and mask have patch_size
    img, mask = random_crop(img, mask, patch_size)
    img, mask = pad_if_needed(img, mask, patch_size)
    return img, mask


def keypoint_scale_crop(
    img: ndarray, mask: ndarray, patch_size: tuple, keypoint: tuple, scale_limit: tuple = (1, 1)
) -> tuple:
    """
    Scale img and mask by a random scale factor in the range of scale_limit. Then Crop img and mask
    to the patch_size around the keypoint and pad if needed to have the desired patch_size.
    """
    # Conversion is needed to be consistent to albumentations notation
    scale_limit = (scale_limit[0] + 1, scale_limit[1] + 1)
    # Randomly scale img, mask and keypoint
    scale_factor = random.uniform(*scale_limit)
    img, mask, keypoint = keypoint_scale(img, mask, scale_factor, keypoint)
    # Keypoint crop img and mask and pad_if_needed afterward to ensure img and mask have patch_size
    img, mask = keypoint_crop(img, mask, patch_size, keypoint)
    img, mask = pad_if_needed(img, mask, patch_size)
    return img, mask
