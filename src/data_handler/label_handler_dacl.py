import os
import json

import numpy as np
import cv2
import torch

from torchvision.transforms.functional import resize
from src.data_handler.label_handler_multilabel import MultiLabelSegmentationHandler

cv2.setNumThreads(0)

TARGET_LIST = [
    "Crack",
    "ACrack",
    "Wetspot",
    "Efflorescence",
    "Rust",
    "Rockpocket",
    "Hollowareas",
    "Cavity",  #
    "Spalling",
    "Graffiti",
    "Weathering",
    "Restformwork",
    "ExposedRebars",
    "Bearing",
    "EJoint",
    "Drainage",
    "PEquipment",
    "JTape",
    "WConccor",
]


class MultiLabelSegmentationHandlerDACL(MultiLabelSegmentationHandler):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.all_pred_json = []

    """
    Needed Prediction Writer
    """

    def save_prediction(self, pred_logits, file):
        t0 = 0.01  # lowest threshold
        t1 = 0.5  # medium threshold
        t2 = 0.99  # highest threshold
        # treshhold = [t0, t2, t0, t0, t0, t0, t1, t0, t0, t1, t0, t0, t0, t1, t1, t1, t1, t1, t1]
        # treshhold = [t0, t2, t0, t0, t0, t1, t1, t0, t0, t1, t0, t0, t1, t0, t1, t0, t1, t1, t1]
        treshhold = [t0, t2, t0, t0, t0, t1, t1, t0, t0, t1, t0, t0, t1, t0, t1, t0, t1, t1, t1]

        treshhold = torch.tensor(treshhold)[:, None, None]
        # Submission have to be for resolution 512x512
        if pred_logits.shape[1] != 512 and pred_logits.shape[2] != 512:
            pred_logits = resize(pred_logits, (512, 512))

        # prediction = (torch.sigmoid(pred_logits.float()) >= 0.5).float()
        prediction = (torch.sigmoid(pred_logits.float()) >= treshhold).float()
        prediction = prediction.numpy()
        name = os.path.split(file)[-1]

        pred_json = {}
        pred_json["imageName"] = name + ".jpg"
        pred_json["imageWidth"] = prediction.shape[1]
        pred_json["imageHeight"] = prediction.shape[2]
        shapes = []
        for i, p in enumerate(prediction):
            class_label = TARGET_LIST[i]
            class_contours, _ = cv2.findContours(
                p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for class_contour in class_contours:
                class_contour = class_contour.squeeze(1)
                if len(class_contour) == 1:
                    continue
                polygon = {
                    "label": class_label,
                    "points": class_contour.astype(np.float32).tolist(),
                    # np.flip(class_contour).astype(np.float32).tolist()
                    "shape_type": "polygon",
                }
                shapes.append(polygon)

        pred_json["shapes"] = shapes
        self.all_pred_json.append(pred_json)
        with open(file + ".json", "w") as f:
            json.dump(pred_json, f)

    def collect_results(self, output_dir):
        with open(os.path.join(output_dir, "predictions.jsonl"), "w") as file:
            for d in self.all_pred_json:
                json.dump(d, file)
                file.write("\n")
