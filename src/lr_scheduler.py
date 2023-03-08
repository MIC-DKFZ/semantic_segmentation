import torch
import warnings
from torch.optim.lr_scheduler import _LRScheduler


def polynomial_LR_scheduler_stepwise(optimizer, max_steps, exponent=0.9, **kwargs):
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (1 - step / max_steps) ** exponent
    )

    return lr_scheduler


def polynomial_LR_scheduler_epochwise(optimizer, max_epochs, exponent=0.9, **kwargs):
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (1 - epoch / (max_epochs)) ** exponent
    )

    return lr_scheduler


class PolynomialLR_Warmstart(_LRScheduler):
    def __init__(
        self, optimizer, warmstart_iters=1, total_iters=5, power=1.0, last_epoch=-1, verbose=False
    ):
        if isinstance(warmstart_iters, int):
            self.warmstart = warmstart_iters
        if isinstance(warmstart_iters, float):
            self.warmstart = int(total_iters * warmstart_iters)
        self.warmstart = max(self.warmstart, 1)

        self.total_iters = total_iters - self.warmstart

        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        # Warmstart
        if self.last_epoch < self.warmstart:
            # addrates = [(lr / (self.warmstart + 1)) for lr in self.base_lrs]
            # updated_lr = [
            #     addrates[i] * (self.last_epoch + 1)
            #     for i, group in enumerate(self.optimizer.param_groups)
            # ]
            addrates = [(lr / (self.warmstart)) for lr in self.base_lrs]
            updated_lr = [
                addrates[i] * (self.last_epoch + 1)
                for i, group in enumerate(self.optimizer.param_groups)
            ]
            return updated_lr

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = (
            (1.0 - self.last_epoch / self.total_iters)
            / (1.0 - (self.last_epoch - 1) / self.total_iters)
        ) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                base_lr
                * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]


