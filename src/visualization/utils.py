from PIL import Image
import torch
import numpy as np
import numpy.typing as npt
import cv2
from typing import Any


def blend_img_mask(img_viz, mask_viz, alpha):
    fig = cv2.addWeighted(img_viz, 1 - alpha, mask_viz, alpha, 0.0)

    bg_map = np.all(mask_viz == [255, 255, 255], axis=2)
    fig[bg_map] = img_viz[bg_map]
    bg_map = np.all(mask_viz == [0, 0, 0], axis=2)
    fig[bg_map] = img_viz[bg_map]
    return fig


def convert_torch_to(img_torch: torch.Tensor, output_type: str) -> Any:
    """
    Converting a torch.tensor to a np.array, PIL.Image or torch.tensor

    Parameters
    ----------
    img_torch: torch.Tensor
    output_type: str

    Returns
    -------
    Any
    """
    if output_type == "numpy":
        return np.array(img_torch)
    elif output_type == "PIL":
        return Image.fromarray(np.array(img_torch))
    elif output_type == "torch":
        return img_torch


def convert_numpy_to(img_np: np.ndarray, output_type: str) -> Any:
    """
    Converting a np.array to a np.array, PIL.Image or torch.tensor

    Parameters
    ----------
    img_np: np.array
    output_type: str

    Returns
    -------
    Any
    """
    if output_type == "numpy":
        return img_np
    elif output_type == "PIL":
        return Image.fromarray(img_np)
    elif output_type == "torch":
        return torch.tensor(img_np)


def show_img(
    img: torch.Tensor,
    mean: list = None,
    std: list = None,
    output_type: str = "numpy",
    *args,
    **kwargs
) -> Any:
    """
    Visualize Images
    Create visualization of a tensor by undoing the normalization and convert to RGB (scaling to
    0-255 with 3 channels)

    Parameters
    ----------
    img: torch.Tensor
    mean: list
    std: list
    output_type: str

    Returns
    -------
    np.ndarry, torch.tensor or PIL.Image, dependent on desired output_type
    """

    img_np = np.array(img)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    else:
        img_np = np.moveaxis(img_np, 0, -1)

    if mean is not None and std is not None:
        img_np = ((img_np * std) + mean) * 255
    elif np.min(img_np) >= 0 and np.max(img_np) <= 1:
        img_np = img_np * 255
    img_np = img_np.astype(np.uint8)

    return convert_numpy_to(img_np, output_type)


def show_mask_sem_seg(
    mask: torch.Tensor, cmap: npt.ArrayLike, output_type: str = "numpy", *args, **kwargs
) -> Any:
    """
    Visualize a Semantic Segmentation mask
    Create visualization of a tensor by colour encode classes with the given cmap
    Convert it to the desired output type

    Parameters
    ----------
    mask : torch.Tensor
    cmap : Any
        list like (np.array,torch.tensor) with len(num_classes) and encoding each class to a RBB value (0-255)
    output_type : str

    Returns
    -------
    np.ndarry, torch.tensor or PIL.Image, dependent on desired output_type
    """
    mask_np = np.array(mask)
    w, h = mask_np.shape
    fig = np.zeros((w, h, 3), dtype=np.uint8)
    for class_id in np.unique(mask_np):
        x, y = np.where(mask_np == class_id)
        if class_id >= len(cmap):
            fig[x, y] = [0, 0, 0]
        else:
            fig[x, y, :] = cmap[class_id]
    fig = fig.astype(np.uint8)
    return convert_numpy_to(fig, output_type)


def show_mask_inst_seg(
    target: dict, cmap=None, output_type: str = "numpy", alpha: float = 0.5, *args, **kwargs
) -> Any:
    """
    Visualize an Instance Segmentation mask
    Create visualization of a tensor by colour encode instance with a random colour
    Convert it to the desired output type

    Parameters
    ----------
    target : dict
    img_shape : list
    output_type : str
    alpha: float
        alpha of the segmentation masks, to see overlapping masks

    Returns
    -------
    np.ndarry, torch.tensor or PIL.Image, dependent on desired output_type
    """
    if len(target["masks"]) == 0:
        return None

    masks = target["masks"].squeeze(1)
    boxes = target["boxes"]
    labels = target["labels"]
    fig = np.ones((*masks.shape[1:], 3), dtype=np.uint8) * 255
    for mask, box, label in zip(masks, boxes, labels):
        # color = np.random.randint(0, 255, 3)
        color = np.random.randint(0, 255, 3) if cmap is None else cmap[label]

        # If also the bounding box should be shown
        # x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # cv2.rectangle(img, (x1, y1), (x2, y2), [int(color[0]), int(color[1]), int(color[2])])

        x, y = np.where(mask != 0)
        fig[x, y] = fig[x, y] * alpha + color * (1 - alpha)
    fig = fig.astype(np.uint8)
    return convert_numpy_to(fig, output_type)


def show_mask_multilabel_seg(
    masks, cmap: npt.ArrayLike, output_type: str = "numpy", alpha=0.5, *args, **kwargs
):
    """
    Visualize a multilabel Semantic Segmentation mask
    Create visualization of a tensor by colour encode classes with the given cmap
    Convert it to the desired output type

    Parameters
    ----------
    mask : torch.Tensor
    cmap : list
        list with len(num_classes) and encoding each class to a RBB value (0-255)
    output_type : str
    alpha: float
        alpha of the segmentation masks, to see overlapping masks

    Returns
    -------
    np.ndarry, torch.tensor or PIL.Image, dependent on desired output_type
    """

    masks_np = np.array(masks)
    c, w, h = masks_np.shape
    fig = np.zeros((w, h, 3), dtype=np.uint8)
    for i, mask in enumerate(masks):
        color = np.array(cmap[i])
        x, y = np.where(mask != 0)
        fig[x, y] = fig[x, y] * alpha + color * (1 - alpha)
    fig = fig.astype(np.uint8)
    return convert_numpy_to(fig, output_type)


def show_prediction_inst_seg(pred, cmap=None, output_type="numpy", alpha=0.5, *args, **kwargs):
    """
    Visualize an Instance Segmentation prediction
    Create visualization of a tensor by colour encode instance (confidence score >=0.5)(Softmax values
    >=0.5) with a random colour
    Convert it to the desired output type

    Parameters
    ----------
    target : dict
    img_shape : list
    output_type : str
    alpha: float
        alpha of the segmentation masks, to see overlapping masks

    Returns
    -------
    np.ndarry, torch.tensor or PIL.Image, dependent on desired output_type
    """
    pred = pred
    masks = pred["masks"].squeeze(1)
    boxes = pred["boxes"]
    scores = pred["scores"]
    labels = pred["labels"]

    fig = np.ones((*masks.shape[1:], 3), dtype=np.uint8) * 255

    masks = [mask for mask, score in zip(masks, scores) if score >= 0.5]
    boxes = [box for box, score in zip(boxes, scores) if score >= 0.5]
    labels = [label for label, score in zip(labels, scores) if score >= 0.5]

    for mask, box, label in zip(masks, boxes, labels):

        # color = np.random.randint(0, 255, 3)
        color = np.random.randint(0, 255, 3) if cmap is None else cmap[label]
        # If also the bounding box should be shown
        # x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # cv2.rectangle(img, (x1, y1), (x2, y2), [int(color[0]), int(color[1]), int(color[2])])

        x, y = np.where(mask >= 0.5)
        fig[x, y] = fig[x, y] * alpha + color * (1 - alpha)
    fig = fig.astype(np.uint8)
    return convert_numpy_to(fig, output_type)
