from typing import List, Union, Tuple, Any

import numpy as np
import torch.nn.functional as F
import cv2
from matplotlib import cm
from numpy import ndarray
from torch import Tensor
import albumentations as A
import torch

from src.visualization.utils import show_mask_sem_seg
from src.data_handler.base_handler import BaseLabelHandler

cv2.setNumThreads(0)


class SemanticSegmentationHandler(BaseLabelHandler):
    def __init__(self, cmap_name="viridis", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cmap = np.array(
            cm.get_cmap(cmap_name, self.num_classes).colors * 255,
            dtype=np.uint8,
        )[:, 0:3]

    def load_file(self, file: str) -> ndarray:
        mask = cv2.imread(file, -1)
        return mask.astype(np.uint8)

    def apply_transforms(
        self, img: ndarray, mask: ndarray, transforms: A.transforms, *args, **kwargs
    ) -> Tuple[Union[ndarray, Any, Tensor], Union[ndarray, Any, Tensor]]:
        if transforms is not None:
            transformed = transforms(image=img, mask=mask, *args, **kwargs)
            img = transformed["image"]
            mask = transformed["mask"]
        return img, mask

    """
    Needed for Sampling 
    """

    def get_class_ids(self, mask: ndarray) -> Union[ndarray, List[int]]:
        return np.unique(mask)

    def get_class_locations(
        self, mask: ndarray, class_id: int
    ) -> Union[Tuple[ndarray, ndarray], Tuple[List[int], List[int]]]:
        x, y = np.where(mask == class_id)
        return x, y

    """
    Needed Prediction Writer
    """

    def to_cpu(self, pred: Tensor) -> Tensor:
        return pred.detach().cpu()

    def save_prediction(self, logits: Tensor, file: str) -> None:
        pred = logits.argmax(0).numpy()

        cv2.imwrite(file + ".png", pred)

    def save_probabilities(self, logits: Tensor, file: str) -> None:
        # TODO, correct dim?
        pred = F.softmax(logits.float(), dim=1)
        pred = pred.numpy()

        np.savez(file + ".npz", probabilities=pred)

    def save_visualization(self, logits: Tensor, file: str) -> None:

        pred = logits.argmax(0).numpy()
        visualization = show_mask_sem_seg(pred, self.cmap, "numpy")
        visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        cv2.imwrite(file + "_viz.png", visualization)

    """
    Needed for Visualization
    """

    def predict_img(self, img, model):
        pred = model(img.unsqueeze(0).cuda())
        return list(pred.values())[0].squeeze().detach().cpu()

    def viz_mask(self, mask: Tensor, *args, **kwargs) -> None:
        return show_mask_sem_seg(mask, self.cmap, *args, **kwargs)

    #
    def viz_prediction(self, logits: Tensor, *args, **kwargs) -> None:
        pred = logits.argmax(0).numpy()
        return show_mask_sem_seg(pred, self.cmap, *args, **kwargs)
