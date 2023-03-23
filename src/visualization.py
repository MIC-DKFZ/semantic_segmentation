from PIL import Image
import torch
import numpy as np
import cv2

from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

from src.utils import has_not_empty_attr, get_logger

log = get_logger(__name__)


def convert_torch_to(img_torch, output_type):
    if output_type == "numpy":
        return np.array(img_torch)
    elif output_type == "PIL":
        return Image.fromarray(np.array(img_torch))
    elif output_type == "torch":
        return img_torch


def convert_numpy_to(img_np, output_type):
    if output_type == "numpy":
        return img_np
    elif output_type == "PIL":
        return Image.fromarray(img_np)
    elif output_type == "torch":
        return torch.tensor(img_np)


def show_img(img: torch.Tensor, mean: list = None, std: list = None, output_type: str = "numpy"):

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


def show_mask_sem_seg(mask: torch.Tensor, cmap: list, output_type: str = "numpy"):
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


def show_mask_inst_seg(target, img_shape, output_type: str = "numpy", alpha=0.5):
    if len(target["masks"]) == 0:
        fig = np.ones((*img_shape, 3), dtype=np.uint8) * 255
        return convert_numpy_to(fig, output_type)

    masks = target["masks"].squeeze(1)
    boxes = target["boxes"]  # .detach().cpu()
    fig = np.ones((*img_shape, 3), dtype=np.uint8) * 255
    for mask, box in zip(masks, boxes):
        color = np.random.randint(0, 255, 3)

        # If also the bounding box should be shown
        # x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # cv2.rectangle(img, (x1, y1), (x2, y2), [int(color[0]), int(color[1]), int(color[2])])

        x, y = np.where(mask != 0)
        fig[x, y] = fig[x, y] * alpha + color * (1 - alpha)
    fig = fig.astype(np.uint8)
    return convert_numpy_to(fig, output_type)


def show_prediction_sem_seg():
    pass


def show_prediction_inst_seg(pred, img_shape, output_type="numpy", alpha=0.5):
    # pred = [{k: v.detach().cpu() for k, v in t.items()} for t in pred]
    # pred = list(p.detach().cpu() for p in pred)
    pred = pred[0]
    masks = pred["masks"].squeeze(1)
    boxes = pred["boxes"]
    scores = pred["scores"]

    masks = [mask for mask, score in zip(masks, scores) if score >= 0.5]
    boxes = [box for box, score in zip(boxes, scores) if score >= 0.5]

    fig = np.ones((*img_shape, 3), dtype=np.uint8) * 255
    for mask, box in zip(masks, boxes):

        color = np.random.randint(0, 255, 3)

        # If also the bounding box should be shown
        # x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # cv2.rectangle(img, (x1, y1), (x2, y2), [int(color[0]), int(color[1]), int(color[2])])

        x, y = np.where(mask >= 0.5)
        fig[x, y] = fig[x, y] * alpha + color * (1 - alpha)
    fig = fig.astype(np.uint8)
    return convert_numpy_to(fig, output_type)


class Visualizer:
    def __init__(
        self,
        dataset: Dataset,
        cmap: np.ndarray,
        model: LightningModule = None,
        mean: list = None,
        std: list = None,
        segmentation: str = "semantic",
        axis: int = 1,
    ) -> None:
        """
        Visualizing a Dataset
        If a model if Given also the prediction is of the model on the dataset is shown

        Parameters
        ----------
        dataset: Dataset
            dataset which should be visualized
        cmap: np.ndarray
            colormap to color the singel classes, list of RGB values
        model: LightningModule, optional
            if given the model is used to generate predictions for images in dataset
        mean: list, optional
            if given the normalization is inverted during visualization --> nicer image
        std: list, optional
            if given the normalization is inverted during visualization --> nicer image
        """
        self.model = model
        self.dataset = dataset
        self.cmap = cmap
        self.mean = mean
        self.std = std
        self.segmentation = segmentation
        self.axis = axis

    def color_mask(self, mask: torch.Tensor, img_shape) -> np.ndarray:
        """
        Color encode mask with color ids into RGB

        Parameters
        ----------
        mask_np
            array of shape [w,h], with class ids for each pixel
        Returns
        -------
        np.ndarray :
            array of shape [w,h,3] with color encoding of each class (in RGB format)
        """
        if self.segmentation == "semantic":
            return show_mask_sem_seg(mask, self.cmap, "numpy")
        elif self.segmentation == "instance":
            return show_mask_inst_seg(mask, img_shape, "numpy")

    def viz_correctness(self, pred: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """
        visualizing the correctness of the prediction (where pred is qual to mask)

        Parameters
        ----------
        pred : torch.Tensor
        mask: torch.Tensor

        Returns
        -------
        np.ndarray :
        """
        cor = np.zeros(self.mask_np.shape, dtype=np.uint8)
        # where prediction and gt are equal
        x, y = np.where(pred == mask)
        # pixel which dont belong to a class (ignore index)
        x_ign, y_ign = np.where(mask > len(self.cmap))

        cor[:, :] = [255, 0, 0]  # Red for not equal pixel
        cor[x, y] = [0, 255, 0]  # Green for equal pixel
        cor[x_ign, y_ign] = [0, 0, 0]  # Black for ignored pixel
        return cor

    def update_window(self, *arg, **kwargs) -> None:
        """
        Update the opencv Window when another image should be displayed (another img_id)
        Load Image and Mask and transform them into the correct format (opencv conform)
        (Optional) if a model is given also predict the image and colorize prediction
        """
        img_id = cv2.getTrackbarPos("Image ID", "Window")

        # Load Image and Mask, transform image and colorize the mask
        img, mask = self.dataset[img_id]

        self.img_np = show_img(img, self.mean, self.std, "numpy")
        if self.segmentation == "semantic":
            self.mask_np = show_mask_sem_seg(mask, self.cmap, "numpy")
        elif self.segmentation == "instance":
            self.mask_np = show_mask_inst_seg(mask, img.shape[-2:], "numpy")

        # Predict the Image and colorize the prediction
        if self.model is not None:
            if self.segmentation == "semantic":
                pred = self.model(img.unsqueeze(0).cuda())
                pred = torch.argmax(list(pred.values())[0].squeeze(), dim=0).detach().cpu()
                self.pred = self.color_mask(np.array(pred), img_shape=img.shape[-2:])

                # Show Correctness of prediction
                self.cor = self.viz_correctness(pred, mask)
            elif self.segmentation == "instance":
                pred = self.model(img.unsqueeze(0).cuda())[0]
                pred = [{k: v.detach().cpu() for k, v in pred.items()}]
                self.pred = show_prediction_inst_seg(pred, img_shape=img.shape[-2:])

        # update the the channel and alpha parameter and show the window
        self.update_channel_and_alpha()

    def update_channel_and_alpha(self, *arg, **kwargs) -> None:
        """
        Select the correct Channel
        if -1 the channels 0:3 are used
        otherwise a single channel is used in grayscale on 3 channels
        """
        if hasattr(self, "img_np"):
            channel_id = cv2.getTrackbarPos("Channel", "Window")

            # Select the correct Channel, if -1 use the channels 0:3 otherwise use a single one
            # and transform to RGB
            if channel_id == -1:
                self.img_np_chan = self.img_np[:, :, 0:3]
            else:
                self.img_np_chan = self.img_np[:, :, channel_id]
                self.img_np_chan = cv2.cvtColor(self.img_np_chan, cv2.COLOR_GRAY2RGB)

            # Update Alpha and udate Window
            self.update_alpha()

    def update_alpha(self, *arg, **kwargs) -> None:
        """
        Display the image blended with the mask or prediction on the left, on the right the gt mask
        Alpha defines the weight of the blending
        Afterwards update the opencv image
        """
        if hasattr(self, "img_np_chan"):
            alpha = cv2.getTrackbarPos("alpha", "Window") / 100

            # Blend the image with prediction
            if hasattr(self, "pred"):
                self.img_np_fig = cv2.addWeighted(
                    self.img_np_chan, 1 - alpha, self.pred, alpha, 0.0
                )
                self.img_np_fig = self.update_corrects(self.img_np_fig)
            else:
                self.img_np_fig = cv2.addWeighted(
                    self.img_np_chan, 1 - alpha, self.mask_np, alpha, 0.0
                )
                bg_map = np.all(self.mask_np == [255, 255, 255], axis=2)
                self.img_np_fig[bg_map] = self.img_np_fig[bg_map]
            # concat blended image and mask
            fig = np.concatenate((self.img_np_fig, self.mask_np), self.axis)
            # transform from RGB to BGR to match the cv2 order
            self.fig = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)
            # show image
            cv2.imshow("Window", self.fig)

    def update_corrects(self, img) -> None:
        """
        Display the image blended with the mask or prediction on the left, on the right the gt mask
        Alpha defines the weight of the blending
        Afterwards update the opencv image
        """
        alpha_cor = cv2.getTrackbarPos("correctness", "Window") / 100
        if alpha_cor > 0:

            img = cv2.addWeighted(img, 1 - alpha_cor, self.cor, alpha_cor, 0.0)
        return img
