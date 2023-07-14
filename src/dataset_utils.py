import numpy as np
from typing import Tuple
from numpy import ndarray


def random_crop(
    img: ndarray,
    mask: ndarray,
    patch_size: Tuple = (600, 600),
):
    """
    Random Crop the Image and Mask Pair to shape (s1, s2, 3) for the image and (s1, s2) for the mask
    with si=max(patch_size[i], img.shape[i]) for i in 1,2

    Parameters
    ----------
    img: ndarray
        input image, should be RGB with shape (with,height,3)
    mask: ndarray
        input mask, should be single channel with shape (with,height)
    patch_size: Tuple, optional
        Size of the output patch

    Returns
    -------
    ndarray
        cropped image
    ndarray
        cropped mask
    """
    w, h, _ = img.shape

    # Get a random point
    x_min = np.random.randint(0, min(1, w - patch_size[0]))
    y_min = np.random.randint(0, min(1, h - patch_size[1]))

    # Get the max value
    x_max = int(min(x_min + patch_size[0], w))
    y_max = int(min(y_min + patch_size[1], h))
    # Copping the image
    img_cropped = img[x_min:x_max, y_min:y_max, :]
    mask_cropped = mask[x_min:x_max, y_min:y_max]
    # Pad if needed
    # ...

    return img_cropped, mask_cropped


def center_crop(
    img: ndarray,
    mask: ndarray,
    x: int,
    y: int,
    patch_size: Tuple = (600, 600),
    offset: float = 0.5,
):
    """
    Center Crop the Image and Mask Pair to shape (s1, s2, 3) for the image and (s1, s2) for the mask
    with si=max(patch_size[i], img.shape[i]) for i in 1,2 and the center(x,y)

    Parameters
    ----------
    img: ndarray
        input image, should be RGB with shape (with,height,3)
    mask: ndarray
        input mask, should be single channel with shape (with,height)
    patch_size: Tuple, optional
        Size of the output patch
    x: int
        y coordinate of the center point around the patch is cropped
    y: int
        y coordinate of the center point around the patch is cropped
    offset: float, optional
        ratio of the patch size which defines the range of the shift in x and y direction

    Returns
    -------
    ndarray
        cropped image
    ndarray
        cropped mask
    """
    w, h, _ = img.shape

    # Clip the patch to be inside the image with min=0 and max=(w,h) - patch_size
    x_min = int(max(min(x - np.ceil(patch_size[0] / 2), w - patch_size[0]), 0))
    y_min = int(max(min(y - np.ceil(patch_size[1] / 2), h - patch_size[1]), 0))

    # Compute
    x_max = int(min(x_min + patch_size[0], w))
    y_max = int(min(y_min + patch_size[1], h))

    # Copping the image
    img_cropped = img[x_min:x_max, y_min:y_max, :]
    mask_cropped = mask[x_min:x_max, y_min:y_max]

    # Pad if needed
    # ...

    return img_cropped, mask_cropped


def split_by_ID(files, ids=None, num_folds=5):
    if ids is None:
        ids = files
    splits = []
    for i in range(0, num_folds):
        val_ids = ids[i::num_folds]
        train_ids = [ID for ID in ids if ID not in val_ids]

        val_samples = [file for file in files if any(val_id in file for val_id in val_ids)]
        train_samples = [file for file in files if any(train_id in file for train_id in train_ids)]
        print(f"Split {i}, #Train: {len(train_samples)}, #Val: {len(val_samples)}")
        splits.append({"train": train_samples, "val": val_samples})
    return splits
