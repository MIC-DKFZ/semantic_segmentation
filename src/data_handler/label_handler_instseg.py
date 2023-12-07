import numpy as np
import torch.nn.functional as F
import cv2
from matplotlib import cm
import torch
from src.visualization.utils import show_mask_sem_seg
from typing import List, Union, Tuple, Any, Dict
from numpy import ndarray
from src.data_handler.base_handler import BaseLabelHandler
from torch import Tensor
from src.visualization.utils import show_prediction_inst_seg, show_mask_inst_seg

cv2.setNumThreads(0)


class InstanceSegmentationHandler(BaseLabelHandler):
    def __init__(self, cmap_name="viridis", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cmap = np.array(
            cm.get_cmap(cmap_name, self.num_classes).colors * 255,
            dtype=np.uint8,
        )[:, 0:3]

    def load_file(self, file):
        mask = cv2.imread(file, -1)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        bboxes = self.get_bbox(masks)
        # area = self.get_area(bboxes)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        labels = torch.ones((len(masks),), dtype=torch.int64)
        return {"boxes": bboxes, "labels": labels, "masks": masks, "area": area}
        # return masks

    def get_bbox(self, masks):
        boxes = []
        for mask in masks:
            pos = np.where(mask)
            xmin = int(np.min(pos[1]))
            xmax = int(np.max(pos[1]))
            ymin = int(np.min(pos[0]))
            ymax = int(np.max(pos[0]))
            boxes.append([xmin, ymin, xmax, ymax])
        return torch.as_tensor(boxes, dtype=torch.float32)

    def apply_transforms(self, img, mask, transforms, *args, **kwargs):
        mask = mask["masks"]
        if transforms is not None:
            # Need to Catch empty masks
            empty_mask = len(mask) == 0

            mask = mask if empty_mask else mask.transpose((1, 2, 0))
            transformed = transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

            mask = mask if empty_mask else mask.permute(2, 0, 1)

        bboxes = self.get_bbox(mask)
        # area = self.get_area(bboxes)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        labels = torch.ones((len(mask),), dtype=torch.int64)
        return img, {"boxes": bboxes, "labels": labels, "masks": mask, "area": area}

    def get_bbox(self, masks):
        boxes = []
        for mask in masks:
            pos = np.where(mask)
            xmin = int(np.min(pos[1]))
            xmax = int(np.max(pos[1]))
            ymin = int(np.min(pos[0]))
            ymax = int(np.max(pos[0]))
            boxes.append([xmin, ymin, xmax, ymax])
        return torch.as_tensor(boxes, dtype=torch.float32)

    def get_class_ids(self, mask: ndarray) -> Union[ndarray, List[int]]:

        return np.unique(mask["labels"])

    def get_class_locations(
        self, mask: ndarray, class_id: int
    ) -> Union[Tuple[ndarray, ndarray], Tuple[List[int], List[int]]]:

        x, y = np.where(mask == class_id)
        return x, y

    """
    Needed Prediction Writer
    """

    def to_cpu(self, pred: Tensor) -> Dict[str, Tensor]:
        return {k: v.detach().cpu() for k, v in pred.items()}

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

    def viz_mask(self, mask: Dict[str, Tensor], *args, **kwargs) -> None:
        return show_mask_inst_seg(mask, self.cmap, *args, **kwargs)

    #
    def viz_prediction(self, logits: Tensor, *args, **kwargs) -> None:
        return show_prediction_inst_seg(logits, self.cmap, *args, **kwargs)
