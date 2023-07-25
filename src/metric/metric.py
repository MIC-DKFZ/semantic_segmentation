import hydra
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection


class MetricModule(MetricCollection):
    def __init__(self, config: DictConfig, **kwargs) -> None:
        """
        Init MetricModule as subclass of MetricCollection
        Can directly be initialized with the config

        Parameters
        ----------
        config: DictConfig
        kwargs :
            arguments passed to super class (MetricCollection)
        """

        metrics = {}
        for name, m_conf in config.items():
            if m_conf is not None:
                metric = hydra.utils.instantiate(m_conf)
                metrics[name] = metric
        super().__init__(metrics, **kwargs)

    # def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    #     if self.call_stepwise:
    #         x = super().forward(*args, **kwargs)
    #         self.log(x)
    #     elif self.call_per_image:
    #         x = super().forward(*args, **kwargs)
    #         self.log(x)
    #     elif self.call_global:
    #         self.update(*args, **kwargs)
    #
    # def compute(self) -> Dict[str, Any]:
    #     if self.call_stepwise:
    #         res = torch.nanmean()
    #     elif self.call_per_image:
    #         res = torch.nanmean()
    #     elif self.call_global:
    #         res = super().compute()
    #     return res
