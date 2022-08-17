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


######### Experimental Stuff #########
class LR_Stepper:
    def __init__(
        self,
        epochs,
        gpus,
        acc_grad_batches,
        coarse_size,
        base_size,
        batch_size,
        start_portion,
        max_coarse_epoch,
    ):

        self.max_epochs = epochs
        # gpus = num_gpus
        # acc_grad_batches =1 #cfg.accumulate_grad_batches
        # coarse_size=cfg.DATASET.SIZE.COARSE
        # base_size=cfg.DATASET.SIZE.TRAIN
        # batch_size=cfg.batch_size
        # start_portion = cfg.DATASET.COARSE_DATA.START_PORTION
        # max_coarse_epoch = cfg.DATASET.COARSE_DATA.MAX_COARSE_EPOCH

        policy = lambda current_epoch: start_portion * np.maximum(
            1 - current_epoch / (max_coarse_epoch - 1), 0
        )

        epochs = np.array(range(self.max_epochs))
        data_size = (policy(epochs) * coarse_size).astype(int) + base_size
        size_dataloader = data_size // batch_size
        steps_per_gpu = (np.ceil(size_dataloader / gpus)).astype(int)

        self.acc_steps_per_gpu = (np.ceil(steps_per_gpu / acc_grad_batches)).astype(int)
        self.cum_steps = np.cumsum(self.acc_steps_per_gpu)
        self.cum_steps = np.insert(self.cum_steps, 0, 0)

    def __getitem__(self, step):
        # step=max(0,step-1)
        epoch = np.argmax(self.cum_steps > step) - 1
        max_step = self.acc_steps_per_gpu[epoch]
        cur_step = step - self.cum_steps[epoch]

        return (1 - (epoch + cur_step / max_step) / (self.max_epochs)) ** 0.9


def polynomial_LR_scheduler_coarse_step(
    optimizer,
    epochs,
    gpus,
    acc_grad_batches,
    coarse_size,
    base_size,
    batch_size,
    start_portion,
    max_coarse_epoch,
    **kwargs
):
    lambda_class = LR_Stepper(
        epochs,
        gpus,
        acc_grad_batches,
        coarse_size,
        base_size,
        batch_size,
        start_portion,
        max_coarse_epoch,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lambda_class[step])
    return lr_scheduler  # step
