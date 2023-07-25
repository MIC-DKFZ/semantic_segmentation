import hydra
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from matplotlib import cm

from src.metric.metric import MetricModule
from src.loss_function import get_loss_function_from_cfg
from src.utils import get_logger, has_not_empty_attr, first_from_dict
from src.visualization import show_mask_sem_seg

log = get_logger(__name__)


class SegModel(LightningModule):
    def __init__(self, model_config: DictConfig) -> None:
        """
        __init__ the LightningModule
        instantiate the model and the metric(s)

        Parameters
        ----------
        config : omegaconf.DictConfig
        """
        super().__init__()
        # config = model_config
        self.config = model_config

        # instantiate model from config
        self.model = hydra.utils.instantiate(self.config.model)
        # self.model = torch.compile(self.model)
        # instantiate metric from config and metric related parameters
        self.metric_name = self.config.METRIC.NAME
        # When to Call the Metric
        self.metric_call_global = self.config.METRIC.call_global
        self.metric_call_stepwise = self.config.METRIC.call_stepwise
        self.metric_call_per_img = self.config.METRIC.call_per_img
        # instantiate validation metric from config and save best_metric parameter
        self.metric = MetricModule(self.config.METRIC.METRICS)  # ,persistent=False)
        self.register_buffer("best_metric_val", torch.as_tensor(0), persistent=False)

        # instantiate train metric from config and save best_metric parameter if wanted
        self.train_metric = self.config.METRIC.train_metric
        if self.train_metric:
            self.metric_train = self.metric.clone()
            self.register_buffer("best_metric_train", torch.as_tensor(0), persistent=False)

        # create colormap for visualizing the example predictions and also define number of example predictions
        self.cmap = torch.tensor(
            cm.get_cmap("viridis", self.config.DATASET.NUM_CLASSES).colors * 255,
            dtype=torch.uint8,
        )[:, 0:3]
        self.num_example_predictions = (
            self.config.num_example_predictions
            if has_not_empty_attr(self.config, "num_example_predictions")
            else 0
        )
        self.viz_mean = (
            self.config.augmentation_cfg.mean
            if has_not_empty_attr(self.config.augmentation_cfg, "mean")
            else None
        )
        self.viz_std = (
            self.config.augmentation_cfg.std
            if has_not_empty_attr(self.config.augmentation_cfg, "std")
            else None
        )

    def configure_optimizers(self) -> dict:
        """
        Instantiate the lossfunction + lossweights from the config
        Instantiate the optimizer from the config
        Instantiate the lr scheduler form the config

        Returns
        -------
        dict :
            contains the optimizer and the scheduler + config

        """

        # instantiate lossfunction and lossweight for each element in list
        if isinstance(self.config.lossfunction, str):
            self.loss_functions = [
                get_loss_function_from_cfg(self.config.lossfunction, self.config)
            ]
        else:
            self.loss_functions = [
                get_loss_function_from_cfg(LF, self.config) for LF in self.config.lossfunction
            ]

        if hasattr(self.config, "lossweight"):
            self.loss_weights = self.config.lossweight
        else:
            self.loss_weights = [1] * len(self.loss_functions)
        log.info(
            "Loss Functions with Weights: %s",
            list(zip(self.loss_functions, self.loss_weights)),
        )

        # instantiate optimizer
        self.optimizer = hydra.utils.instantiate(self.config.optimizer, self.parameters())

        # instantiate lr scheduler
        max_steps = self.trainer.datamodule.max_steps()

        lr_scheduler_config = dict(self.config.lr_scheduler)
        lr_scheduler_config["scheduler"] = hydra.utils.instantiate(
            self.config.lr_scheduler.scheduler,
            optimizer=self.optimizer,
            # max_steps=max_steps,
            total_iters=max_steps,
        )

        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, x: torch.Tensor) -> dict:
        """
        forward the input to the model
        if model prediction is not a dict covert the output into a dict

        Parameters
        ----------
        x : torch.Tensor
            input to predict

        Returns
        -------
        dict of {str:torch.Tensor} :
            prediction of the model which a separate key for each model output
        """
        x = self.model(x)
        # covert output to dict if output is list, tuple or tensor
        if not isinstance(x, dict):
            if isinstance(x, list) or isinstance(x, tuple):
                keys = ["out" + str(i) for i in range(len(x))]
                x = dict(zip(keys, x))
            elif isinstance(x, torch.Tensor):
                x = {"out": x}
        # if torch.isnan(x["out"]).any():
        #     print("NAN Predicted")
        return x

    def forward_tta(
        self, x: torch.Tensor, scales: list = [1], flip: bool = False, binary_flip: bool = False
    ):
        """
        forward the input to the model and use test time augmentation (tta)
        scaling and flipping is used for augmentation (both optional)

        Parameters
        ----------
        x : torch.Tensor
            input to predict
        scales: list, optional
            List of scales for tta
        flip: bool, optional
            If True the image is flipped 4x and the predictions are averages
        binary_flip: bool, optional
            If True the image is flipped 2x(left and right) and the predictions are averages

        Returns
        -------
        dict of {str:torch.Tensor} :
            prediction of the model
        """
        x_size = x.size(2), x.size(3)
        total_pred = None

        # Iterate through the scales and sum the predictions up
        for scale in scales:
            s_size = int(x_size[0] * scale), int(x_size[1] * scale)

            # Scale input to the target scale
            if scale == 1:
                x_scaled = x.clone()
            else:
                x_scaled = F.interpolate(x, s_size, mode="bilinear", align_corners=True)

            # Prediction of current scaled image
            y_prediction = first_from_dict(self(x_scaled))

            # Flip the image and take the average of all predicctions
            if flip:
                # Take all 4 possible flips
                y_prediction += torch.flip(
                    first_from_dict(self(torch.flip(x_scaled.clone(), [2]))), [2]
                )
                y_prediction += torch.flip(
                    first_from_dict(self(torch.flip(x_scaled.clone(), [3]))), [3]
                )
                y_prediction += torch.flip(
                    first_from_dict(self(torch.flip(x_scaled.clone(), [2, 3]))), [2, 3]
                )
                y_prediction /= 4

            elif binary_flip:
                # Only take the left and right flip
                y_prediction += torch.flip(
                    first_from_dict(self(torch.flip(x_scaled.clone(), [3]))), [3]
                )
                y_prediction /= 2

            # Scale prediction back to the original scale
            if scale != 1:
                y_prediction = F.interpolate(
                    y_prediction, x_size, mode="bilinear", align_corners=True
                )

            # Summing the predictions up
            if total_pred == None:
                total_pred = y_prediction
            else:
                total_pred += y_prediction

        # Average the prediction over all scales
        total_pred /= len(scales)

        return {"out": total_pred}

    def training_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        """
        Forward the image through the model and compute the loss
        (optional) update the metric stepwise of global (defined by metric_call parameter)

        Parameters
        ----------
        batch : list of torch.Tensor
            contains img (shape==[batch_size,num_classes,w,h]) and mask (shape==[batch_size,w,h])
        batch_idx : int
            index of the batch

        Returns
        -------
        torch.Tensor :
            training loss
        """
        # predict batch
        x, y_gt = batch
        y_pred = self(x)

        # compute and log loss
        loss = self.get_loss(y_pred, y_gt)
        self.log(
            name="Loss/training_loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        # (optional) update train metric
        if self.train_metric:
            self.update_metric(
                list(y_pred.values())[0], y_gt, self.metric_train, prefix="metric_train/"
            )

        return loss

    def validation_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        """
        Forward the image through the model and compute the loss
        update the metric stepwise of global (defined by metric_call parameter)

        Parameters
        ----------
        batch : list of torch.Tensor
            contains img (shape==[batch_size,num_classes,w,h]) and mask (shape==[batch_size,w,h])
        batch_idx : int
            index of the batch

        Returns
        -------
        torch.Tensor :
            validation loss
        """

        # predict batch
        x, y_gt = batch
        y_pred = self(x)

        # # compute and log loss to tensorboard
        val_loss = self.get_loss(y_pred, y_gt)
        self.log(
            name="Loss/validation_loss",
            value=val_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        # update validation metric
        # print(first_from_dict(y_pred).dtype, y_gt.dtype)
        self.update_metric(first_from_dict(y_pred), y_gt, self.metric, prefix="metric/")

        # log some example predictions to tensorboard
        if self.global_rank == 0 and not self.trainer.sanity_checking:
            self.log_batch_prediction(x, first_from_dict(y_pred), y_gt, batch_idx)

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metric to logger and console
        Reset the validation metric
        """
        if not self.trainer.sanity_checking:
            log.info("EPOCH: %s", self.current_epoch)

            # compute and log global validation metric to tensorboard
            # if self.metric_call in ["global", "global_and_stepwise"]:
            if self.metric_call_global:
                metric = self.metric.compute()
                self.log_dict_epoch(metric, prefix="metric/", on_step=False, on_epoch=True)

            # log validation metric to console
            self.metric_logger(
                metric_group="metric/",
                best_metric="best_metric_val",
                stage="Validation",
                save_metric_state=True,
            )

        # reset metric manually
        self.metric.reset()

    def on_train_epoch_end(self) -> None:
        """
        (optional) Log the training metric to logger and console
        (optional) Reset the training metric
        """
        # (optional) compute and log global validation metric to tensorboard
        if self.train_metric:
            if self.metric_call_global:
                metric_train = self.metric_train.compute()

                # log train metric to tensorboard
                self.log_dict_epoch(
                    metric_train, prefix="metric_train/", on_step=False, on_epoch=True
                )
                # reset metric manually

            # log train metric to console
            self.metric_logger(
                metric_group="metric_train/",
                best_metric="best_metric_train",
                stage="Train",
                save_metric_state=False,
            )
            self.metric_train.reset()

    def on_test_start(self) -> None:
        """
        Set the different scales, if no ms testing is used only scale 1 is used
        if not defined also no flipping is done
        """
        # Configure Test Loss
        ignore_index = (
            self.config.DATASET.IGNORE_INDEX
            if has_not_empty_attr(self.config.DATASET, "IGNORE_INDEX")
            else -100
        )
        self.loss_function_test = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Configure TTA behaviour during Testing
        # Scaling
        if has_not_empty_attr(self.config, "TESTING") and has_not_empty_attr(
            self.config.TESTING, "SCALES"
        ):
            self.test_scales = self.config.TESTING.SCALES
        else:
            self.test_scales = [1]
        # Flipping x4
        if has_not_empty_attr(self.config, "TESTING") and has_not_empty_attr(
            self.config.TESTING, "FLIP"
        ):
            self.test_flip = self.config.TESTING.FLIP
        else:
            self.test_flip = False
        # Flipping x2
        if has_not_empty_attr(self.config, "TESTING") and has_not_empty_attr(
            self.config.TESTING, "BINARY_FLIP"
        ):
            self.test_binary_flip = self.config.TESTING.BINARY_FLIP
        else:
            self.test_binary_flip = False

    def test_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        """
        For each scale used during testing:
            resize the image to the desired scale
            forward it to the model
            resize to original size
            (optional) flip the the input image and repeat the above steps
            summing up the prediction
        update the metric stepwise of global (defined by metric_call parameter)

        Parameters
        ----------
        batch : list of torch.Tensor
            contains img (shape==[batch_size,num_classes,w,h]) and mask (shape==[batch_size,w,h])
        batch_idx : int
            index of the batch

        Returns
        -------
        torch.Tensor :
            validation loss
        """
        x, y_gt = batch
        y_pred = self.forward_tta(x, self.test_scales, self.test_flip, self.test_binary_flip)

        # update the metric with the aggregated prediction
        self.update_metric(first_from_dict(y_pred), y_gt, self.metric, prefix="metric_test/")

        # compute and return loss of final prediction
        test_loss = self.loss_function_test(first_from_dict(y_pred), y_gt.long())

        self.log(
            "Loss/Test_loss",
            test_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.trainer.num_devices > 1 else False,
        )

        # log some example predictions to tensorboard
        if self.global_rank == 0:
            self.log_batch_prediction(x, first_from_dict(y_pred), y_gt, batch_idx)

        return test_loss

    def on_test_epoch_end(self) -> None:
        # compute the metric and log the metric
        log.info("TEST RESULTS")

        # compute and log global validation metric to tensorboard
        metric = self.metric.compute()
        self.log_dict_epoch(metric, prefix="metric_test/", on_step=False, on_epoch=True)

        # log validation metric to console
        self.metric_logger(
            metric_group="metric_test/",
            best_metric="best_metric_val",
            stage="Test",
            save_metric_state=True,
        )

        # reset metric manually
        self.metric.reset()

    def update_metric(
        self, y_pred: torch.Tensor, y_gt: torch.Tensor, metric: MetricModule, prefix: str = ""
    ):

        if self.metric_call_stepwise:
            # Log the metric result for each step
            metric_step = metric(y_pred, y_gt)
            # exclude nan since pl uses torch.mean for reduction, this way torch.nanmean is simulated
            metric_step = {k: v for k, v in metric_step.items() if not torch.isnan(v)}
            self.log_dict_epoch(
                metric_step,
                prefix=prefix,
                postfix="_stepwise",
                on_step=False,
                on_epoch=True,
            )
        elif self.metric_call_per_img:
            # If metric should be called per img, iterate through the batch to compute and log the
            # metric for each img separately
            for yi_pred, yi_gt in zip(y_pred, y_gt):
                metric_step = metric(yi_pred.unsqueeze(0), yi_gt.unsqueeze(0))
                # exclude nan since pl uses torch.mean for reduction, this way torch.nanmean is simulated
                metric_step = {k: v for k, v in metric_step.items() if not torch.isnan(v)}
                self.log_dict_epoch(
                    metric_step,
                    prefix=prefix,
                    postfix="_per_img",
                    on_step=False,
                    on_epoch=True,
                )
        elif self.metric_call_global:
            # Just update the metric
            metric.update(y_pred, y_gt)

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
        # loss_f = torch.nn.CrossEntropyLoss()
        # loss = loss_f(first_from_dict(y_pred), y_gt)
        # return loss
        if self.training:
            loss = sum(
                [
                    self.loss_functions[i](y, y_gt.long()) * self.loss_weights[i]
                    for i, y in enumerate(y_pred.values())
                ]
            )
        else:
            loss = sum(
                [
                    F.cross_entropy(
                        y,
                        y_gt.long(),
                        ignore_index=self.config.DATASET.IGNORE_INDEX
                        if has_not_empty_attr(self.config.DATASET, "IGNORE_INDEX")
                        else -100,
                    )
                    * self.loss_weights[i]
                    for i, y in enumerate(y_pred.values())
                ]
            )
        return loss

    def metric_logger(self, metric_group, best_metric, stage="Validation", save_metric_state=False):
        """
        logging the metric by:
        update the best metric
        (optional) save state dict variables if provided by the metrics (currently not used)
        log best metric to tensorboard
        log target metric and best metric to console
        log remaining metrics to console

        Parameters
        ----------
        metric_group : str
            enables to group parameters in tensorboard, e.g. into metric/
        best_metric : str
            name of the best metric which corresponds to the target metric
        stage : str, optional
            Current stage, needed for nicer logging
        save_matric_state : bool, optional
            if the metric_state should be saved, currently not used
        """
        logged_metrics = self.trainer.logged_metrics
        metrics = {
            k.replace(metric_group, ""): v for k, v in logged_metrics.items() if metric_group in k
        }

        # update best target metric
        target_metric_score = metrics.pop(self.metric_name)
        if target_metric_score > getattr(self, best_metric):
            setattr(self, best_metric, target_metric_score)

        # (optional) save state of metrics if wanted and provided by the metric, only on rank 0
        if save_metric_state and self.global_rank == 0:
            for name, met in self.metric.items():
                if hasattr(met, "save_state"):
                    met.save_state(self.trainer)

        # log best metric to tensorboard
        if "best_" + self.metric_name in metrics:
            metrics.pop("best_" + self.metric_name)
        self.log_dict_epoch(
            {self.metric_name: getattr(self, best_metric)},
            prefix=metric_group + "best_",
        )
        # log target metric and best metric to console
        log.info(
            stage.ljust(10) + " - Best %s %.4f       %s: %.4f",
            self.metric_name,
            getattr(self, best_metric),
            self.metric_name,
            target_metric_score,
        )
        # remove best_metric from metrics since best metric is already logged to console
        if "best_" + self.metric_name in metrics:
            metrics.pop("best_" + self.metric_name)

        # log remaining metrics to console
        for name, score in metrics.items():
            log.info("%s: %.4f", name, score)

    def log_dict_epoch(self, dic: dict, prefix: str = "", postfix: str = "", **kwargs) -> None:
        """
        Logging a dict to Tensorboard logger but instead of the current step use the current epoch
        on the x-axis of the graphs

        Parameters
        ----------
        dic : dict of {str, torch.Tensor}
            each item contains the name and the scalar for the parameter to log
        prefix : str, optional
            prefix which is added to the name of all logged parameters
        postfix: str, optional
            postfix which is added to the name of all logged parameters
        kwargs: optional
            Parameters to pass to self.log_dict
        """
        for name, score in dic.items():
            self.log_dict(
                {
                    prefix + name + postfix: score,
                    "step": torch.tensor(self.current_epoch, dtype=torch.float32),
                },
                logger=True,
                sync_dist=True if self.trainer.num_devices > 1 else False,
                **kwargs,
            )

    def viz_label(self, label, cmap, output_type):
        label = label.cpu()
        return show_mask_sem_seg(label, cmap, output_type)

    def viz_prediction(self, pred, cmap, output_type, treshhold=0.5):
        pred = pred.argmax(0).detach().cpu()
        return show_mask_sem_seg(pred, cmap, output_type)

    def log_batch_prediction(
        self, imgs: torch.Tensor, preds: torch.Tensor, gts: torch.Tensor, batch_idx: int
    ) -> None:
        """
        logging example prediction and gt to tensorboard

        Parameters
        ----------
        img: torch.Tensor
        pred : torch.Tensor
        gt : torch.Tensor
        batch_idx: int
            idx of the current batch, needed for naming of the predictions
        """
        # Check if the current batch has to be logged, if yes how many images
        val_batch_size = self.trainer.datamodule.val_batch_size
        diff_to_show = self.num_example_predictions - (batch_idx * val_batch_size)
        if diff_to_show > 0:
            current_batche_size = len(imgs)
            # log the desired number of images
            for i in range(min(current_batche_size, diff_to_show)):
                pred = self.viz_prediction(preds[i], self.cmap, "torch")
                gt = self.viz_label(gts[i], self.cmap, "torch")

                # Concat pred and gt for better visualization
                axis = 0 if gt.shape[1] > 2 * gt.shape[0] else 1
                fig = torch.cat((pred, gt), axis)

                # Resize Figure to reduce size of tensorboard files
                w, h, c = fig.shape
                max_size = 1024
                if max(w, h) > max_size:
                    s = max_size / max(w, h)
                    fig = fig.permute(2, 0, 1).unsqueeze(0).float()
                    fig = F.interpolate(fig, size=(int(w * s), int(h * s)), mode="nearest")
                    fig = fig.squeeze(0).permute(1, 2, 0)

                # Log Figure to tensorboard
                self.trainer.logger.experiment.add_image(
                    "Example_Prediction/prediction_gt__sample_"
                    + str(batch_idx * val_batch_size + i),
                    fig.to(torch.uint8),
                    self.current_epoch,
                    dataformats="HWC",
                )
