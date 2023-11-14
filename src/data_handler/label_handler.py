from abc import ABC, abstractmethod
import numpy as np
import albumentations as A
import cv2
import torch
from matplotlib import cm

from src.visualization.visualizer import (
    show_mask_multilabel_seg,
)

cv2.setNumThreads(0)


class BaseHandler(ABC):
    @abstractmethod
    def load_mask(self, file):
        pass

    @abstractmethod
    def save_mask(self, mask, name):
        pass

    @abstractmethod
    def save_prediction(self, pred_logit, name):
        pass

    @abstractmethod
    def save_probabilities(self, pred_logit, name):
        pass

    @abstractmethod
    def apply_transforms(self, img, mask, transforms):
        pass

    @abstractmethod
    def get_class_ids(self, mask):
        pass

    @abstractmethod
    def to_cpu(self, pred):
        pass

    @abstractmethod
    def get_class_locations(self, mask, class_id):
        pass

    @abstractmethod
    def viz_mask(self, mask, img):
        pass

    @abstractmethod
    def viz_prediction(self, pred, img):
        pass

    @abstractmethod
    def get_from_batch(self, batch, id):
        pass


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
        prediction = (torch.sigmoid(pred_logits) >= 0.5).float().numpy()
        for i, pred in enumerate(prediction):
            cv2.imwrite(f"{file}_{i}.png", pred)

    def save_probabilities(self, pred_logits, file):

        pred = torch.sigmoid(pred_logits)
        pred = pred.numpy()

        np.savez(file + ".npz", probabilities=pred)

    def save_visualization(self, pred_logits, file):
        color_map = "viridis"
        cmap = np.array(cm.get_cmap(color_map, self.num_classes).colors * 255, dtype=np.uint8)[
            :, 0:3
        ]

        pred = (torch.sigmoid(pred_logits) >= 0.5).numpy()
        visualization = show_mask_multilabel_seg(pred, cmap, "numpy")
        cv2.imwrite(file + "_viz.png", visualization)


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

    def apply_transforms(self, img, mask, transforms):
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


if __name__ == "__main__":
    import random

    # class RandomCrop2(A.DualTransform):
    #     """Crop a random part of the input.
    #
    #     Args:
    #         height (int): height of the crop.
    #         width (int): width of the crop.
    #         p (float): probability of applying the transform. Default: 1.
    #
    #     Targets:
    #         image, mask, bboxes, keypoints
    #
    #     Image types:
    #         uint8, float32
    #     """
    #
    #     def __init__(self, height, width, always_apply=False, p=1.0):
    #         super().__init__(always_apply, p)
    #         self.height = height
    #         self.width = width
    #
    #     def apply_with_params(self, params, **kwargs):  # skipcq: PYL-W0613
    #         print(params)
    #         print(kwargs.keys())
    #
    #         if params is None:
    #             return kwargs
    #         params = self.update_params(params, **kwargs)
    #         print(params)
    #         res = {}
    #         for key, arg in kwargs.items():
    #             if arg is not None:
    #                 target_function = self._get_target_function(key)
    #                 target_dependencies = {
    #                     k: kwargs[k] for k in self.target_dependence.get(key, [])
    #                 }
    #                 res[key] = target_function(arg, **dict(params, **target_dependencies))
    #             else:
    #                 res[key] = None
    #         return res
    #
    #     def apply(self, img, h_start=0, w_start=0, **params):
    #         return F.random_crop(img, self.height, self.width, h_start, w_start)
    #
    #     def get_params(self):
    #         return {"h_start": random.random(), "w_start": random.random()}
    #
    #     def apply_to_bbox(self, bbox, **params):
    #         return F.bbox_random_crop(bbox, self.height, self.width, **params)
    #
    #     def apply_to_keypoint(self, keypoint, **params):
    #         return F.keypoint_random_crop(keypoint, self.height, self.width, **params)
    #
    #     def get_transform_init_args_names(self):
    #
    #         return ("height", "width")

    class KeypointCrop(A.DualTransform):
        def __init__(self, height, width, always_apply=False, p=1.0):
            super(KeypointCrop, self).__init__(always_apply, p)
            self.height = height
            self.width = width

        def apply_with_params(self, params, **kwargs):

            keypoint = kwargs["keypoints"][0]
            image = kwargs["image"]
            mask = kwargs["mask"]

            w, h, _ = img.shape

            # Clip the patch to be inside the image with min=0 and max=(w,h) - patch_size
            x_min = int(max(min(keypoint[0] - np.ceil(self.height / 2), w - self.height), 0))
            y_min = int(max(min(keypoint[1] - np.ceil(self.width / 2), h - self.width), 0))

            # Compute
            x_max = int(min(x_min + self.height, w))
            y_max = int(min(y_min + self.width, h))

            # Copping the image
            image = image[x_min:x_max, y_min:y_max]
            mask = mask[x_min:x_max, y_min:y_max]
            res = {
                "image": image,
                "mask": mask,
                "keypoints": [(keypoint[0] - x_min, keypoint[1] - y_min, 0, 0)],
            }
            return res

        # def apply(self, img, **params):
        #     return F.crop(
        #         img, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max
        #     )
        #
        # def apply_to_bbox(self, bbox, **params):
        #     return F.bbox_crop(
        #         bbox,
        #         x_min=self.x_min,
        #         y_min=self.y_min,
        #         x_max=self.x_max,
        #         y_max=self.y_max,
        #         **params,
        #     )
        #
        # def apply_to_keypoint(self, keypoint, **params):
        #     return F.crop_keypoint_by_coords(
        #         keypoint, crop_coords=(self.x_min, self.y_min, self.x_max, self.y_max)
        #     )
        #
        # def get_transform_init_args_names(self):
        #     return ("height", "width")

    img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    mask = np.random.randint(0, 5, (1024, 1024), dtype=np.uint8)
    print(img.shape)
    print(mask.shape)
    # random_scale_crop = A.Compose(
    #     [
    #         A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
    #         KeypointCrop(height=512, width=512),
    #         A.PadIfNeeded(min_height=512, min_width=512),
    #     ]
    # )

    keypoint_scale_crop_A = A.Compose(
        [
            A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
            KeypointCrop(height=512, width=512),
            A.PadIfNeeded(min_height=512, min_width=512),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    # keypoint_scale_crop_B=
    # Apply the transform to the image
    out_1 = keypoint_scale_crop_A(image=img, mask=mask, keypoints=[(512, 512)])
    # out_2 = keypoint_scale_crop_B(image=img, mask=mask)
    # Extract the augmented image from the transformed dictionary
    print(out_1["image"].shape)
    # print(out_2["image"].shape)
    print(out_1["mask"].shape)
    print(out_1["keypoints"])
    # print(out_2["mask"].shape)
