import numpy as np
import torch.nn.functional as F
import cv2
from matplotlib import cm

from src.visualization.utils import show_mask_sem_seg


cv2.setNumThreads(0)


class SemanticSegmentationHandler:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    """
    Basic behaviour needed for Training
    """

    def load_mask(self, file):
        mask = cv2.imread(file, -1)
        return mask

    def apply_transforms(self, img, mask, transforms, *args, **kwargs):
        if transforms is not None:
            transformed = transforms(image=img, mask=mask, *args, **kwargs)
            img = transformed["image"]
            mask = transformed["mask"]
        return img, mask

    """
    Needed for Sampling 
    """

    def get_class_ids(self, mask):
        return np.unique(mask)

    def get_class_locations(self, mask, class_id):
        x, y = np.where(mask == class_id)
        return x, y

    """
    Needed Prediction Writer
    """

    def to_cpu(self, pred):
        return pred.detach().cpu()

    def save_prediction(self, pred_logits, file):
        pred = pred_logits.argmax(0).numpy()

        cv2.imwrite(file + ".png", pred)

    def save_probabilities(self, pred_logits, file):
        # TODO, correct dim?
        pred = F.softmax(pred_logits.float(), dim=1)
        pred = pred.numpy()

        np.savez(file + ".npz", probabilities=pred)

    def save_visualization(self, pred_logits, file):
        color_map = "viridis"
        cmap = np.array(cm.get_cmap(color_map, self.num_classes).colors * 255, dtype=np.uint8)[
            :, 0:3
        ]

        pred = pred_logits.argmax(0).numpy()
        visualization = show_mask_sem_seg(pred, cmap, "numpy")
        visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        cv2.imwrite(file + "_viz.png", visualization)

    """
    Needed for Visualization
    """

    def viz_mask(self, mask, cmap, *args, **kwargs):
        return show_mask_sem_seg(mask, cmap, *args, **kwargs)

    #
    def viz_prediction(self, pred_logits, cmap, *args, **kwargs):
        pred = pred_logits.argmax(0).numpy()
        return show_mask_sem_seg(pred, cmap, *args, **kwargs)

    def predict_img(self, img, model):
        pred = model(img.unsqueeze(0).cuda())
        return list(pred.values())[0].squeeze().detach().cpu()
