from typing import Dict, List, Any, Union
from abc import abstractmethod

import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss
import lightning as L
from torchmetrics import MetricCollection
from matplotlib import cm

from src.metric.metric import MetricModule
from src.loss.configure_loss import get_loss_function_from_cfg
from src.utils.utils import get_logger
from src.utils.config_utils import first_from_dict

log = get_logger(__name__)


class BaseModel(L.LightningModule):
    # TODO include Torchmetric plot()
    # TODO update exclude keys
    # TODO Max steps more elegant
    # TODO Predict Data Loading
    # TODO Predict Multilabel and Instance
    # TODO Ignore BG
    # TODO resize TTA
    # TODO viz in prediction summary writer
    # TODO training.py logging correct output dir
    # Base Trainer for Semantic Segmentation, Multilabel Segmentation and Instance Segmentation
    def __init__(self, cfg: DictConfig) -> None:
        """
        Init the LightningModule
        Config - For a better overview divide the Config into their subparts
        Config - Get some parameters from the config for easy access
        Model - Instantiate torch model from config
        Metric - instantiate validation metric, set best_metric parameter
        Metric - Instantiate train metric from config and save best_metric parameter if wanted
        Lossfunction - Instantiate loss and validation loss

        Parameters
        ----------
        cfg : omegaconf.DictConfig
        """
        super().__init__()

        # Config - For a better overview divide the Config into their subparts
        self.model_cfg: DictConfig = cfg.model
        self.dataset_cfg: DictConfig = cfg.dataset
        self.metric_cfg: DictConfig = cfg.metric
        self.optimizer_cfg: DictConfig = cfg.optimizer
        self.lr_scheduler_cfg: DictConfig = cfg.lr_scheduler
        self.loss_cfg: DictConfig = cfg.loss
        self.tta_cfg: DictConfig = cfg.tta

        # Config - Get some parameters from the config for easy access
        self.num_classes: int = cfg.dataset.num_classes
        self.ignore_index: int = cfg.dataset.ignore_index
        self.num_example_predictions: int = cfg.logging.num_example_predictions

        # Model - Instantiate torch model from config
        self.model: nn.Module = hydra.utils.instantiate(self.model_cfg.model)

        # Metric - instantiate validation metric, set best_metric parameter
        self.metric: MetricCollection = MetricModule(self.metric_cfg.metrics)
        self.register_buffer("best_metric_val", torch.as_tensor(0), persistent=False)

        # Metric - Instantiate train metric from config and save best_metric parameter if wanted
        if self.metric_cfg.train_metric:
            self.metric_train: MetricCollection = self.metric.clone()
            self.register_buffer("best_metric_train", torch.as_tensor(0), persistent=False)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Instantiate the optimizer from the config
        Instantiate the lr scheduler form the config

        Returns
        -------
        Dict[str, Any]
            Contains the optimizer and the scheduler + config

        """
        # Lossfunction - Instantiate loss and validation loss
        self.configure_loss()

        # Instantiate Optimizer
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.optimizer_cfg, self.parameters()
        )

        # Instantiate LR Scheduler and Config
        max_steps: int = self.trainer.datamodule.max_steps()
        scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
            self.lr_scheduler_cfg.scheduler, optimizer=optimizer, total_iters=max_steps
        )
        lr_scheduler_config: dict = dict(self.lr_scheduler_cfg)
        lr_scheduler_config["scheduler"] = scheduler

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def configure_loss(self) -> None:
        """
        Instantiate the lossfunction
        Instantiate the validation/test lossfunction
        Get lossweights from the config
        """
        # Lossfunction - Instantiate for each element in list
        if isinstance(self.loss_cfg.function, str):
            self.loss_cfg.function = [self.loss_cfg.function]
        self.loss_functions: List[_Loss] = [
            get_loss_function_from_cfg(LF, self.dataset_cfg, self.device)
            for LF in self.loss_cfg.function
        ]

        # Lossfunction - Instantiate Validation/Training Lossfunction for each element in list
        if isinstance(self.loss_cfg.val_function, str):
            self.loss_cfg.val_function = [self.loss_cfg.val_function]
        self.val_loss_functions: List[_Loss] = [
            get_loss_function_from_cfg(LF, self.dataset_cfg, self.device)
            for LF in self.loss_cfg.val_function
        ]

        # Lossweight - Get lossweight for each element in list
        if hasattr(self.loss_cfg, "weight"):
            self.loss_weights: List[float] = self.loss_cfg.weight
        else:
            self.loss_weights: List[float] = [1] * len(self.loss_functions)

        log.info(
            f"Loss Functions with Weights: {list(zip(self.loss_functions, self.loss_weights))}",
        )

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def validation_step(self, *args, **kwargs) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metric to logger and console
        Reset the validation metric
        """
        if not self.trainer.sanity_checking:
            log.info("EPOCH: %s", self.current_epoch)

            # Metric - compute and log global validation metric to tensorboard
            if self.metric_cfg.metric_global:
                metric = self.metric.compute()
                self.log_dict_epoch(metric, prefix="metric/", on_step=False, on_epoch=True)

            # Logging - log validation metric to console
            self.log_metric(
                metric_group="metric/",
                best_metric="best_metric_val",
                stage="Validation",
                save_metric_state=True,
            )

        # Metric - reset manually
        self.metric.reset()

    def on_train_epoch_end(self) -> None:
        """
        (Optional) Log the training metric to logger and console
        (Optional) Reset the training metric
        """
        # (optional) compute and log global validation metric to tensorboard
        if self.metric_cfg.train_metric:
            if self.metric_cfg.metric_global:

                # Metric - compute and log global validation metric to tensorboard
                metric_train = self.metric_train.compute()
                self.log_dict_epoch(
                    metric_train, prefix="metric_train/", on_step=False, on_epoch=True
                )

            # Logging - log train metric to console
            self.log_metric(
                metric_group="metric_train/",
                best_metric="best_metric_train",
                stage="Train",
                save_metric_state=False,
            )

            # Metric - reset manually
            self.metric_train.reset()

    def on_test_epoch_start(self) -> None:
        # Lossfunction - Initialize the lossfunction for testing, since configure_optimizers is
        # not called during trainer.test(...)
        self.configure_loss()

    @abstractmethod
    def test_step(self, *args, **kwargs) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        """
        Log the test metric to logger and console
        Reset the test metric
        """
        log.info("TEST RESULTS")

        # Metric - compute and log global test metric to tensorboard
        if self.metric_cfg.metric_global:
            metric = self.metric.compute()
            self.log_dict_epoch(metric, prefix="metric_test/", on_step=False, on_epoch=True)

        # Logging - log test metric to console
        self.log_metric(
            metric_group="metric_test/",
            best_metric="best_metric_val",
            stage="Test",
            save_metric_state=True,
        )

        # Metric - reset manually
        self.metric.reset()

    @abstractmethod
    def predict_step(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_loss(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def update_metric(self, *args, **kwargs) -> None:
        pass

    def log_metric(
        self,
        metric_group: str,
        best_metric: str,
        stage: str = "Validation",
        save_metric_state: bool = False,
    ) -> None:
        """
        Get all metrics which are logged
        Update best target metric
        (Optional) Save state of metrics
        Log metric to logger and console

        Parameters
        ----------
        metric_group : str
            enables to group parameters in tensorboard, e.g. into metric/
        best_metric : str
            name of the best metric which corresponds to the target metric
        stage : str, optional
            Current stage, needed for nicer logging
        save_metric_state : bool, optional
            if the metric_state should be saved, currently not used
        """
        # Name of the target metric
        metric_name: str = self.metric_cfg.name

        # Get all metrics which are logged into the metric group and remove prefix
        logged_metrics: dict = self.trainer.logged_metrics
        metrics: dict = {
            k.replace(metric_group, ""): v for k, v in logged_metrics.items() if metric_group in k
        }

        # Update best target metric
        if metrics[metric_name] > getattr(self, best_metric):
            setattr(self, best_metric, metrics[metric_name])
        metrics["best_" + metric_name] = getattr(self, best_metric)

        # (Optional) Save state of metrics if wanted and provided by the metric, only on rank 0
        if save_metric_state and self.global_rank == 0:
            for name, met in self.metric.items():
                if hasattr(met, "save_state"):
                    met.save_state(self.trainer)

        # Get Best and Current metric and remove them from the metrics dict (for separate logging)
        current_metric = metrics.pop(metric_name)
        best_metric = metrics.pop("best_" + metric_name)

        # Log the best metric at the current epoch to logger
        self.log_dict_epoch(
            {metric_name: best_metric},
            prefix=metric_group + "best_",
        )

        # Log Best and Current metric separately to console
        log.info(
            f"{stage.ljust(10)} {metric_name} - "
            f"Best: {best_metric:.4f}     "
            f"Current: {current_metric:.4f}"
        )

        # Log remaining metrics to console
        for name, score in metrics.items():
            log.info(f"{name}: {score:.4f}")

    def log_dict_epoch(
        self, dic: Dict[str, Any], prefix: str = "", postfix: str = "", **kwargs
    ) -> None:
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

    @abstractmethod
    def viz_data(self, *args, **kwargs) -> torch.Tensor:
        pass

    def log_batch_prediction(
        self,
        imgs: Union[Dict[str, torch.Tensor], torch.Tensor],
        preds: Union[Dict[str, torch.Tensor], torch.Tensor],
        gts: Union[Dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Logging example prediction and gt to tensorboard

        Parameters
        ----------
        imgs: Union[Dict[str : torch.Tensor], torch.Tensor]
        preds : Union[Dict[str : torch.Tensor], torch.Tensor]
        gts : Union[Dict[str : torch.Tensor], torch.Tensor]
        batch_idx: int
            idx of the current batch, needed for naming of the predictions
        """
        # Check if the current batch has to be logged, if yes how many images
        val_batch_size = self.trainer.datamodule.val_batch_size
        diff_to_show = self.num_example_predictions - (batch_idx * val_batch_size)
        if diff_to_show > 0:
            if isinstance(preds, dict):
                preds = first_from_dict(preds)
            if isinstance(imgs, dict):
                imgs = first_from_dict(imgs)
            if isinstance(gts, dict):
                gts = first_from_dict(gts)

            current_batche_size = len(imgs)

            cmap = torch.tensor(
                cm.get_cmap("viridis", self.num_classes).colors * 255,
                dtype=torch.uint8,
            )[:, 0:3]
            # Log the desired number of images
            for i in range(min(current_batche_size, diff_to_show)):
                fig = self.viz_data(imgs[i], preds[i], gts[i], cmap, "torch")

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
