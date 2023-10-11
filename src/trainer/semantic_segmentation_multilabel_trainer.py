from typing import Dict

import torch

from src.trainer.semantic_segmentation_trainer import SegModel
from src.utils.utils import get_logger
from src.utils.visualization import show_mask_multilabel_seg
import cv2

cv2.setNumThreads(0)
log = get_logger(__name__)


class SegMLModel(SegModel):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = super().forward(x)

        # pred["out"] = torch.sigmoid(pred["out"])
        return pred

    def viz_data(
        self,
        img: torch.Tensor,
        pred: torch.Tensor,
        gt: torch.Tensor,
        cmap: torch.Tensor,
        output_type: str,
    ) -> torch.Tensor:
        """
        Visualize the Data for logging
        In this Case Prediction and GT are visualized and appended

        Parameters
        ----------
        img: torch.Tensor
        pred: torch.Tensor
        gt: torch.Tensor
        cmap: torch.Tensor
        output_type: str

        Returns
        -------
        torch.Tensor

        """
        threshold = 0.5
        pred = torch.sigmoid(pred)
        pred = (pred >= threshold).float().detach().cpu()
        gt = gt.cpu()

        gt = show_mask_multilabel_seg(gt, cmap, output_type)
        pred = show_mask_multilabel_seg(pred, cmap, output_type)

        axis = 0 if gt.shape[1] > 2 * gt.shape[0] else 1
        fig = torch.cat((pred, gt), axis)
        return fig
