import torch
import numpy as np


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
