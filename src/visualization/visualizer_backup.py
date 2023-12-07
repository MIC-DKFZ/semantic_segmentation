import hydra
import numpy as np
import cv2
from lightning import LightningModule
from torch.utils.data import Dataset
from omegaconf.dictconfig import DictConfig
from src.visualization.utils import blend_img_mask
from src.utils.utils import get_logger

log = get_logger(__name__)


class Visualizer:
    def __init__(
        self,
        dataset: Dataset,
        model: LightningModule = None,
        mean: list = None,
        std: list = None,
        image_loader=None,
        label_handler=None,
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
        segmentation: str
            which type of segmentation is used, semantic, multilabel or instance
        axis: int
            show images side by side (1) or above each other (0)
        """
        self.model = model
        self.dataset = dataset
        self.mean = mean
        self.std = std
        # self.segmentation = segmentation
        # Setup Data Handlers
        self.img_handler = (
            hydra.utils.instantiate(image_loader)
            if isinstance(image_loader, DictConfig)
            else image_loader
        )
        self.label_handler = (
            hydra.utils.instantiate(label_handler)
            if isinstance(label_handler, DictConfig)
            else label_handler
        )
        self.axis = axis

    def update_window(self, *arg, **kwargs) -> None:
        """
        Update the opencv Window when another image should be displayed (another img_id)
        Load Image and Mask and transform them into the correct format (opencv conform)
        (Optional) if a model is given also predict the image and colorize prediction
        """
        img_id = cv2.getTrackbarPos("Image ID", "Window")

        # Load Image and Mask, transform image and colorize the mask
        img, mask = self.dataset[img_id][:2]

        self.img_np = self.img_handler.show_img(img, self.mean, self.std, output_type="numpy")
        self.mask_np = self.label_handler.viz_mask(mask, output_type="numpy")

        # Predict the Image and colorize the prediction
        if self.model is not None:
            pred = self.label_handler.infer_img(img, self.model)
            self.pred = self.label_handler.viz_prediction(
                pred, img_shape=img.shape[-2:], output_type="numpy"
            )

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
                self.img_np_fig = blend_img_mask(self.img_np_chan, self.pred, alpha)

            else:
                self.img_np_fig = blend_img_mask(self.img_np_chan, self.mask_np, alpha)

            # concat blended image and mask
            fig = np.concatenate((self.img_np_fig, self.mask_np), self.axis)
            # transform from RGB to BGR to match the cv2 order
            self.fig = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)
            # show image
            cv2.imshow("Window", self.fig)
