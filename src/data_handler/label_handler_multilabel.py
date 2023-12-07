from typing import List, Union, Tuple, Any
import numpy as np
from numpy import ndarray
from torch import Tensor
import albumentations as A
import cv2
import torch
from matplotlib import cm
import torch.nn as nn
from src.visualization.utils import show_mask_multilabel_seg
from src.utils.config_utils import first_from_dict
from src.data_handler.base_handler import BaseLabelHandler

cv2.setNumThreads(0)


class MultiLabelSegmentationHandler(BaseLabelHandler):
    def __init__(self, cmap_name="viridis", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cmap = np.array(
            cm.get_cmap(cmap_name, self.num_classes).colors * 255,
            dtype=np.uint8,
        )[:, 0:3]

    def get_files(self, path: str) -> List[str]:
        mask_files = super().get_files(path)
        mask_files = [mask.rsplit("_", 1)[0] for mask in mask_files]
        mask_files = np.unique(mask_files)

        return list(sorted(mask_files))

    """
    Basic behaviour needed for Training
    """

    def load_file(self, file: str) -> ndarray:
        masks = []
        for i in range(0, self.num_classes):
            mask = cv2.imread(f"{file}_{i}.png", -1)
            masks.append(mask)
        return np.array(masks, dtype=np.uint8)

    def apply_transforms(
        self, img: ndarray, mask: ndarray, transforms: A.transforms, *args, **kwargs
    ) -> Tuple[Union[ndarray, Any, Tensor], Union[ndarray, Any, Tensor]]:
        if transforms is not None:
            mask = mask.transpose((1, 2, 0))
            transformed = transforms(image=img, mask=mask, *args, **kwargs)
            img = transformed["image"]

            mask = transformed["mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.permute(2, 0, 1)
            else:
                mask = mask.transpose((2, 0, 1))
        return img, mask

    """
    Needed for Sampling 
    """

    def get_class_ids(self, mask: ndarray) -> Union[ndarray, List[int]]:
        return [i for i, m in enumerate(mask) if np.any(m)]

    def get_class_locations(
        self, mask: ndarray, class_id: int
    ) -> Union[Tuple[ndarray, ndarray], Tuple[List[int], List[int]]]:
        x, y = np.where(mask[class_id] == 1)
        return x, y

    """
    Needed Prediction Writer
    """

    def to_cpu(self, pred: Tensor) -> Tensor:
        return pred.detach().cpu()

    def save_prediction(self, logits: Tensor, file: str) -> None:
        prediction = (torch.sigmoid(logits.float()) >= 0.5).float().numpy()
        for i, pred in enumerate(prediction):
            cv2.imwrite(f"{file}_{i}.png", pred)

    def save_probabilities(self, logits: Tensor, file: str) -> None:

        pred = torch.sigmoid(logits.float())
        pred = pred.numpy()

        np.savez(file + ".npz", probabilities=pred)

    def save_visualization(self, logits: Tensor, file: str) -> None:

        pred = (torch.sigmoid(logits.float()) >= 0.5).numpy()
        visualization = show_mask_multilabel_seg(pred, self.cmap, "numpy")
        visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        cv2.imwrite(file + "_viz.png", visualization)

    """
    Needed for Visualization
    """

    def predict_img(self, img: Tensor, model: nn.Module) -> Tensor:
        pred = model(img.unsqueeze(0).cuda())
        return self.to_cpu(first_from_dict(pred).squeeze())

    def viz_mask(self, mask: Tensor, *args, **kwargs) -> None:
        return show_mask_multilabel_seg(mask, self.cmap, *args, **kwargs)

    #
    def viz_prediction(self, logits: Tensor, *args, **kwargs) -> None:
        pred = (torch.sigmoid(logits.float()) >= 0.5).numpy()
        return show_mask_multilabel_seg(pred, self.cmap, *args, **kwargs)
