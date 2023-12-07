import numpy as np
import torch.nn.functional as F
import cv2
from matplotlib import cm
import torch
from src.visualization.utils import show_mask_sem_seg


cv2.setNumThreads(0)


class InstanceSegmentationHandler:
    def load_mask(self, file):
        mask = cv2.imread(file, -1)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        bboxes = self.get_bbox(masks)
        area = self.get_area(bboxes)
        labels = torch.ones((len(masks),), dtype=torch.int64)
        return {"boxes": bboxes, "labels": labels, "masks": masks, "area": area}
        # return masks

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

        bboxes = self.get_bbox()
        area = self.get_area(bboxes)
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
