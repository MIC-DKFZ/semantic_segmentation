import argparse
import os
import glob
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)

import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from matplotlib import cm

from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

from trainers.Semantic_Segmentation_Trainer import SegModel

from src.utils import has_not_empty_attr, get_logger
from datasets.DataModules import get_augmentations_from_config

log = get_logger(__name__)


class Visualizer:
    def __init__(
        self,
        dataset: Dataset,
        cmap: np.ndarray,
        model: LightningModule = None,
        mean: list = None,
        std: list = None,
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

    def color_mask(self, mask_np: np.ndarray) -> np.ndarray:
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
        w, h = mask_np.shape
        fig = np.zeros((w, h, 3), dtype=np.uint8)
        for class_id in np.unique(mask_np):
            x, y = np.where(mask_np == class_id)
            if class_id > len(self.cmap):
                fig[x, y] = [0, 0, 0]
            else:
                fig[x, y, :] = self.cmap[class_id]
        return fig

    def transform_img(self, img: torch.Tensor) -> np.ndarray:
        """
        Transform input tensor to numpy array
        If single channel image convert to three channels (GRAY to RGB)
        Move axis from [3,w,h] to [w,h,3]
        Correct mean and std if given

        Parameters
        ----------
        img: torch.Tensor
            input Tensor with one or three channels
        Returns
        -------
        np.ndarray :
            numpy array in RGB format and shape [w,h,3]
        """
        img_np = np.array(img)

        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            img_np = np.moveaxis(img_np, 0, -1)

        if self.mean is not None and self.std is not None:
            img_np = ((img_np * self.std) + self.mean) * 255

        return img_np.astype(np.uint8)

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
        print(type(pred), type(mask))
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
        self.img_np = self.transform_img(img)
        self.mask_np = self.color_mask(np.array(mask))

        # Predict the Image and colorize the prediction
        if self.model is not None:
            pred = self.model(img.unsqueeze(0).cuda())
            pred = torch.argmax(list(pred.values())[0].squeeze(), dim=0).detach().cpu()
            self.pred = self.color_mask(np.array(pred))

            # Show Correctness of prediction
            self.cor = self.viz_correctness(pred, mask)

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

            # Select the correct Channel, if -1 use the channgels 0:3 otherwise use a single one
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
                img_np = cv2.addWeighted(self.img_np_chan, 1 - alpha, self.pred, alpha, 0.0)
                img_np = self.update_corrects(img_np)
            else:
                img_np = cv2.addWeighted(self.img_np_chan, 1 - alpha, self.mask_np, alpha, 0.0)
            # concat blended image and mask
            fig = np.concatenate((img_np, self.mask_np), 1)
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
            # print(self.pred.shape,self.mask_np.shape,(self.pred[:,:]==self.mask_np[:,:]).shape)
            # x,y=np.where(self.pred[:,:]==self.mask_np[:,:])
            # print(x.shape,y.shape)
            # cor=np.zeros(self.mask_np.shape)
            # cor[:,:]=[255,0,0]
            # cor[x,y]=[0,255,0,]

            # cor=cor.astype(np.uint8)

            img = cv2.addWeighted(img, 1 - alpha_cor, self.cor, alpha_cor, 0.0)
        return img


def show_prediction(overrides_cl: list, augmentation: str, split: str) -> None:
    """
    Show Model Predictions
    Load Model and Dataset from the checkpoint(ckpt_dir)
    Show predictions of the model for the images in a small GUI

    Parameters
    ----------
    overrides_cl : list of strings
        arguments from the commandline to overwrite the config
    """
    # initialize hydra
    hydra.initialize(config_path="../config", version_base="1.1")

    # change working dir to checkpoint dir
    ckpt_dir = None
    for override in overrides_cl:
        if override.startswith("ckpt_dir"):
            ckpt_dir = override.split("=", 1)[1]
            break
    if ckpt_dir is None:
        log.error(
            "ckpt_dir has to be in th config. Run python show_prediction.py ckpt_dir=<some.path>"
        )
        quit()
    os.chdir(ckpt_dir)

    # load overrides from the experiment in the checkpoint dir
    overrides_ckpt = OmegaConf.load(os.path.join("hydra", "overrides.yaml"))
    # compose config by override with overrides_ckpt, afterwards override with overrides_cl
    cfg = hydra.compose(config_name="testing", overrides=overrides_ckpt + overrides_cl)

    # Get the TESTING.OVERRIDES to check if additional parameters should be changed
    if has_not_empty_attr(cfg, "TESTING"):
        if has_not_empty_attr(cfg.TESTING, "OVERRIDES"):
            overrides_test = cfg.TESTING.OVERRIDES
            # Compose config again with including the new overrides
            cfg = hydra.compose(
                config_name="testing",
                overrides=overrides_ckpt + overrides_test + overrides_cl,
            )

    # load the best checkpoint and load the model
    cfg.ORG_CWD = os.getcwd()
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]
    if hasattr(cfg.MODEL, "PRETRAINED"):
        cfg.MODEL.PRETRAINED = False
    model = SegModel.load_from_checkpoint(ckpt_file, config=cfg, strict=False).cuda()
    # model = SegModel.load_from_checkpoint(ckpt_file, config=cfg).cuda()
    # print(cfg)
    # print(cfg.model)

    # model=hydra.utils.instantiate(cfg.model).cuda()

    OmegaConf.set_struct(cfg, False)
    if augmentation == "train":
        transforms = get_augmentations_from_config(cfg.AUGMENTATIONS.TRAIN)[0]
    elif augmentation == "val":
        transforms = get_augmentations_from_config(cfg.AUGMENTATIONS.VALIDATION)[0]
    elif augmentation == "test":
        transforms = get_augmentations_from_config(cfg.AUGMENTATIONS.TEST)[0]
    else:
        transforms = A.Compose([ToTensorV2()])

    # check if data is normalized, if yes redo this during visualization of the image
    mean = None
    std = None
    for t in transforms.transforms:  # .transforms:
        if isinstance(t, A.Normalize):
            mean = t.mean
            std = t.std
            break

    # instantiate dataset
    dataset = hydra.utils.instantiate(cfg.dataset, split=split, transforms=transforms)

    # define colormap
    color_map = "viridis"
    cmap = np.array(cm.get_cmap(color_map, cfg.DATASET.NUM_CLASSES).colors * 255, dtype=np.uint8)[
        :, 0:3
    ]

    # init visualizer
    visualizer = Visualizer(dataset, cmap, model, mean, std)

    # create window
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1200, 1200)

    # Create Trackbar for Image Id and alpha value
    cv2.createTrackbar("Image ID", "Window", 0, len(dataset) - 1, visualizer.update_window)

    cv2.createTrackbar("alpha", "Window", 50, 100, visualizer.update_alpha)
    cv2.createTrackbar("correctness", "Window", 0, 100, visualizer.update_alpha)

    # look at the first image to get the number of channels
    img, _ = dataset[0]
    if len(img.shape) == 2:
        channels = 2
    else:
        channels = img.shape[0]

    # Create the Trackbar for the Channel Parameter
    cv2.createTrackbar("Channel", "Window", -1, channels - 1, visualizer.update_channel_and_alpha)
    cv2.setTrackbarMin("Channel", "Window", -1)
    cv2.setTrackbarPos("Channel", "Window", -1)

    # show the first image in window and start loop
    model.eval()
    with torch.no_grad():
        visualizer.update_window()
        print("press q to quit")
        while True:
            k = cv2.waitKey()
            if k == 113:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation",
        type=str,
        default="test",
        help="Which augmentations to use: train, val or test (by default)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="which split to use: train, val or test (by default)",
    )
    args, overrides = parser.parse_known_args()
    augmentation = args.augmentation
    split = args.split

    show_prediction(overrides, augmentation, split)
