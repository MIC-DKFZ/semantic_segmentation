import os
from os.path import join
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf, _update_n
from src.utils.config_utils import first_from_dict


class SemSegPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="batch", save_probabilities=False):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.save_probabilities = save_probabilities

        os.makedirs(self.output_dir, exist_ok=True)

    def save_softmax(self, pred_sm, name):
        np.savez(join(self.output_dir, name + ".npz"), probabilities=pred_sm)

    def save_prediction(self, pred, name):
        cv2.imwrite(join(self.output_dir, name + ".png"), pred)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        preds, names = prediction
        preds_sm = first_from_dict(preds)
        preds = preds_sm.argmax(1)
        preds_sm = F.softmax(preds_sm, dim=1) if self.save_probabilities else preds_sm

        preds = preds.detach().cpu().numpy()
        preds_sm = preds_sm.detach().cpu().numpy()

        for name, pred, pred_sm in zip(names, preds, preds_sm):
            self.save_prediction(pred, name)
            if self.save_probabilities:
                self.save_softmax(pred_sm, name)


class customModelCheckpoint(ModelCheckpoint):
    """
    Small modification on the ModelCheckpoint from pytorch lightning for renaming the last epoch
    """

    def __init__(self, **kwargs):
        super(
            customModelCheckpoint,
            self,
        ).__init__(**kwargs)
        self.CHECKPOINT_NAME_LAST = self.filename.replace("best_", "last_")


class customTQDMProgressBar(TQDMProgressBar):
    """
    Small modification on the TQDMProgressBar class from pytorch lightning to get rid of the
    "v_num" entry and the printing bug during validation (linebreak + print in every step)
    https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning/66731318#66731318
    https://github.com/PyTorchLightning/pytorch-lightning/issues/765
    this is another solution to use the terminal as output console for Pycharm
    https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status = "None"

    def init_validation_tqdm(self):
        ### disable validation tqdm instead only use train_progress_bar###
        bar = tqdm(
            disable=True,
        )
        return bar

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module) -> None:
        # reset progress, set status and update metric
        self.train_progress_bar.reset(
            convert_inf(self.total_train_batches + self.total_val_batches)
        )
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
        self.status = "Training"
        self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        n = self.total_train_batches + batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["status"] = self.status

        return items

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not trainer.sanity_checking:
            self.status = "Validation"
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.status = "Done"
        self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        self.train_progress_bar.refresh()
        print("")


class TimeCallback(Callback):
    """
    Callback for measuring the time during train, validation and testing
    """

    def __init__(self):
        self.t_train_start = torch.cuda.Event(enable_timing=True)
        self.t_val_start = torch.cuda.Event(enable_timing=True)
        self.t_test_start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.t_train = []
        self.t_val = []

    def on_train_epoch_start(self, *args, **kwargs):
        self.t_train_start.record()

    def on_validation_epoch_start(self, trainer, *args, **kwargs):
        if not trainer.sanity_checking:
            self.end.record()
            torch.cuda.synchronize()

            train_time = self.t_train_start.elapsed_time(self.end) / 1000
            self.t_train.append(train_time)

            self.log(
                "Time/train_time",
                train_time,
                logger=True,
                sync_dist=True if trainer.num_devices > 1 else False,
                prog_bar=True,
            )
            self.log(
                "Time/mTrainTime",
                np.mean(self.t_train),
                logger=True,
                sync_dist=True if trainer.num_devices > 1 else False,
            )

        if not trainer.sanity_checking:
            self.t_val_start.record()

    def on_validation_epoch_end(self, trainer, *args, **kwargs):
        if not trainer.sanity_checking:
            self.end.record()
            torch.cuda.synchronize()

            val_time = self.t_val_start.elapsed_time(self.end) / 1000
            self.t_val.append(val_time)

            self.log(
                "Time/validation_time",
                val_time,
                logger=True,
                sync_dist=True if trainer.num_devices > 1 else False,
                prog_bar=True,
            )
            self.log(
                "Time/mValTime",
                np.mean(self.t_val),
                logger=True,
                sync_dist=True if trainer.num_devices > 1 else False,
            )

    def on_test_epoch_start(self, trainer, *args, **kwargs):
        self.t_test_start.record()

    def on_test_epoch_end(self, trainer, *args, **kwargs):
        self.end.record()
        torch.cuda.synchronize()

        test_time = self.t_test_start.elapsed_time(self.end) / 1000

        self.log(
            "Time/test_time",
            test_time,
            logger=True,
            sync_dist=True if trainer.num_devices > 1 else False,
        )
