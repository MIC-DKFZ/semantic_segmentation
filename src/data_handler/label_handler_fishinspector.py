import numpy as np
import cv2
from src.data_handler.label_handler_multilabel import MultiLabelSegmentationHandler


class MultiLabelSegmentationHandlerFI(MultiLabelSegmentationHandler):
    def load_mask(self, file):
        masks = []
        for i in range(0, self.num_classes):
            mask = cv2.imread(f"{file}_{i:04d}.png", -1)
            masks.append(mask)
        return np.array(masks, dtype=np.uint8)
