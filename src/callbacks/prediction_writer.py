import os
from os.path import join

import cv2
import numpy as np
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter
from matplotlib import cm

from src.utils.config_utils import first_from_dict
from src.utils.visualization import (
    show_img,
    show_mask_multilabel_seg,
    show_mask_sem_seg,
    show_prediction_inst_seg,
    show_mask_inst_seg,
)


class SemSegPredictionWriter(BasePredictionWriter):
    """
    How to save semantic segmentation predictions
    """

    def __init__(
        self,
        output_dir,
        write_interval="batch",
        save_probabilities=False,
        save_visualization=False,
        num_classes=None,
        mean=None,
        std=None,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.save_probabilities = save_probabilities
        self.save_visualization = save_visualization
        self.num_classes = num_classes
        self.mean = mean
        self.std = std

        os.makedirs(self.output_dir, exist_ok=True)

    def save_viz(self, img, pred, name):
        color_map = "viridis"
        cmap = np.array(cm.get_cmap(color_map, self.num_classes).colors * 255, dtype=np.uint8)[
            :, 0:3
        ]
        img_viz = show_img(img, self.mean, self.std, "numpy")
        mask_viz = show_mask_sem_seg(pred, cmap, "numpy")
        fig = 0.5 * img_viz + 0.5 * mask_viz
        cv2.imwrite(join(self.output_dir, name + "_viz.png"), fig)
        # fi = cv2.addWeighted(img, 1 - alpha_cor, self.cor, alpha_cor, 0.0)

    def save_softmax(self, pred_sm, name):
        np.savez(join(self.output_dir, name + ".npz"), probabilities=pred_sm)

    def save_prediction(self, pred, name):
        cv2.imwrite(join(self.output_dir, name + ".png"), pred)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        img, names = batch
        prediction_sm = first_from_dict(prediction)
        prediction = prediction_sm.argmax(1)
        prediction_sm = (
            F.softmax(prediction_sm, dim=1) if self.save_probabilities else prediction_sm
        )

        prediction = prediction.detach().cpu().numpy()
        prediction_sm = prediction_sm.detach().cpu().numpy()

        for name, pred, pred_sm in zip(names, prediction, prediction_sm):
            self.save_prediction(pred, name)
            if self.save_probabilities:
                self.save_softmax(pred_sm, name)
            if self.save_visualization:
                self.save_viz(img, pred, name)


class SemSegMLPredictionWriter(SemSegPredictionWriter):
    """
    How to save multilabel segmentation predictions
    """

    def save_viz(self, img, pred, name):
        color_map = "viridis"
        cmap = np.array(cm.get_cmap(color_map, self.num_classes).colors * 255, dtype=np.uint8)[
            :, 0:3
        ]
        img_viz = show_img(img, self.mean, self.std, "numpy")
        mask_viz = show_mask_multilabel_seg(pred, cmap, "numpy")

        alpha_cor = 0.5
        fig = cv2.addWeighted(img_viz, 1 - alpha_cor, mask_viz, alpha_cor, 0.0)
        # fig[mask_viz == [0, 0, 0]] = img_viz[mask_viz == [0, 0, 0]]
        # fig[mask_viz == [255, 255, 255]] = img_viz[mask_viz == [255, 255, 255]]
        # fig = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)

        cv2.imwrite(join(self.output_dir, name + "_viz.png"), fig)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        img, names = batch
        prediction_sm = first_from_dict(prediction)
        prediction = (prediction_sm >= 0.5).float()
        prediction_sm = (
            F.softmax(prediction_sm, dim=1) if self.save_probabilities else prediction_sm
        )

        prediction = prediction.detach().cpu().numpy()
        prediction_sm = prediction_sm.detach().cpu().numpy()

        for name, pred, pred_sm in zip(names, prediction, prediction_sm):
            for i, p in enumerate(pred):
                self.save_prediction(p, name + f"_{i}")
            if self.save_probabilities:
                self.save_softmax(pred_sm, name)
            if self.save_visualization:
                self.save_viz(img[0].detach().cpu(), pred, name)


class InstSegPredictionWriter(SemSegPredictionWriter):
    """
    How to save instance segmentation predictions
    """

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        imgs, names = batch
        conf_threshold = 0.5
        seg_threshold = 0.5
        for pred, img, name in zip(prediction, imgs, names):
            img_shape = img.shape[-2:]
            sem_seg = np.zeros(img_shape)
            inst_seg = np.zeros(img_shape)

            pred = {k: v.detach().cpu() for k, v in pred.items()}
            masks = pred["masks"].squeeze(1)
            scores = pred["scores"]
            labels = pred["labels"]
            masks = [mask for mask, score in zip(masks, scores) if score >= conf_threshold]
            labels = [label for label, score in zip(labels, scores) if score >= conf_threshold]
            for i, (mask, label) in enumerate(zip(masks, labels)):
                x, y = np.where(mask >= seg_threshold)
                sem_seg[x, y] = label
                inst_seg[x, y] = i + 1
            self.save_prediction(sem_seg, name + "_class")
            self.save_prediction(inst_seg, name + "_instance")

            if self.save_probabilities:
                self.save_softmax(np.array([m.numpy() for m in masks]), name)
