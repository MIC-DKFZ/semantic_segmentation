import torch

from trainers.Semantic_Segmentation_Trainer import SegModel
from src.utils.utils import get_logger
from src.utils.visualization import show_mask_multilabel_seg
import cv2

cv2.setNumThreads(0)
log = get_logger(__name__)


class SegMCModel(SegModel):
    def get_loss(self, y_pred: dict, y_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute Loss of each Output and Lossfunction pair (defined by order in Output dict and
        loss_function list), weight them afterward by the corresponding loss_weight and sum the up
        During Validation only use CE loss for runtime reduction

        Parameters
        ----------
        y_pred : dict of {str: torch.Tensor}
            Output prediction of the network as a dict, where the order inside the dict has to
            match the order of the lossfunction defined in the config
            Shape of each tensor: [batch_size, num_classes, w, h]
        y_gt : torch.Tensor
            The ground truth segmentation mask
            with shape: [batch_size, w, h]

        Returns
        -------
        torch.Tenor
            weighted sum of the losses of the individual model outputs
        """

        loss = sum(
            [
                self.loss_functions[i](y, y_gt.to(torch.float16))
                for i, y in enumerate(y_pred.values())
            ]
        )
        return loss

    def viz_label(self, label, cmap, output_type):
        label = label.cpu()
        return show_mask_multilabel_seg(label, cmap, output_type)

    def viz_prediction(self, pred, cmap, output_type, treshhold=0.5):
        pred = (pred >= treshhold).float().detach().cpu()
        return show_mask_multilabel_seg(pred, cmap, output_type)
