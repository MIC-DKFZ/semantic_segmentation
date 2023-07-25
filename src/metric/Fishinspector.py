import torch
from torchmetrics.functional.classification.confusion_matrix import (
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)

from src.metric.IoU import IoU_MultiLabel


class IoU_MultiLabel_partly_labeled(IoU_MultiLabel):
    def update(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _multilabel_confusion_matrix_tensor_validation(
                preds, targets, self.num_labels, self.ignore_index
            )
        confmat = torch.zeros(
            (self.num_labels, 2, 2), device=self.confmat.device, dtype=self.confmat.dtype
        )
        for pred, target, mask in zip(preds, targets, masks):
            pred, target = _multilabel_confusion_matrix_format(
                pred.unsqueeze(0),
                target.unsqueeze(0),
                self.num_labels,
                self.threshold,
                self.ignore_index,
            )
            conf = _multilabel_confusion_matrix_update(pred, target, self.num_labels)
            confmat[mask] += conf[mask]
        self.confmat += confmat
