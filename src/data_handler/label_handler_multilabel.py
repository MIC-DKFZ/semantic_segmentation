import numpy as np

import cv2
import torch
from matplotlib import cm

from src.visualization.utils import show_mask_multilabel_seg
from src.utils.config_utils import first_from_dict

cv2.setNumThreads(0)


class MultiLabelSegmentationHandler:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    """
    Basic behaviour needed for Training
    """

    def load_mask(self, file):
        masks = []
        for i in range(0, self.num_classes):
            mask = cv2.imread(f"{file}_{i}.png", -1)
            masks.append(mask)
        return np.array(masks, dtype=np.uint8)

    def apply_transforms(self, img, mask, transforms, *args, **kwargs):
        if transforms is not None:
            mask = mask.transpose((1, 2, 0))
            transformed = transforms(image=img, mask=mask, *args, **kwargs)
            img = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1)
        return img, mask

    """
    Needed for Sampling 
    """

    def get_class_ids(self, mask):
        return [i for i, m in enumerate(mask) if np.any(m)]

    def get_class_locations(self, mask, class_id):
        x, y = np.where(mask[class_id] == 1)
        return x, y

    """
    Needed Prediction Writer
    """

    def to_cpu(self, pred):
        return pred.detach().cpu()

    def save_prediction(self, pred_logits, file):
        prediction = (torch.sigmoid(pred_logits.float()) >= 0.5).float().numpy()
        for i, pred in enumerate(prediction):
            cv2.imwrite(f"{file}_{i}.png", pred)

    def save_probabilities(self, pred_logits, file):

        pred = torch.sigmoid(pred_logits.float())
        pred = pred.numpy()

        np.savez(file + ".npz", probabilities=pred)

    def save_visualization(self, pred_logits, file):
        color_map = "viridis"
        cmap = np.array(cm.get_cmap(color_map, self.num_classes).colors * 255, dtype=np.uint8)[
            :, 0:3
        ]

        pred = (torch.sigmoid(pred_logits.float()) >= 0.5).numpy()
        visualization = show_mask_multilabel_seg(pred, cmap, "numpy")
        visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        cv2.imwrite(file + "_viz.png", visualization)

    """
    Needed for Visualization
    """

    def viz_mask(self, mask, cmap, *args, **kwargs):
        return show_mask_multilabel_seg(mask, cmap, *args, **kwargs)

    #
    def viz_prediction(self, pred_logits, cmap, *args, **kwargs):
        pred = (torch.sigmoid(pred_logits.float()) >= 0.5).numpy()
        return show_mask_multilabel_seg(pred, cmap, *args, **kwargs)

    def infer_img(self, img, model):
        pred = model(img.unsqueeze(0).cuda())
        return self.to_cpu(first_from_dict(pred).squeeze())
