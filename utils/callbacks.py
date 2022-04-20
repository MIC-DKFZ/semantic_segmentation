import numpy as np
import torch
from tqdm import tqdm
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import convert_inf, Tqdm
import sys
import time

### small modification on the ModelCheckpoint -> naming the last epoch ###
class customModelCheckpoint(ModelCheckpoint):
    def __init__(self,**kwargs):
        super(customModelCheckpoint,self,).__init__(**kwargs)
        self.CHECKPOINT_NAME_LAST="last_epoch_{epoch}"

### small modification on the TQDMProgressBar class to get rid of the v_num entry and duple printing during validation###
### https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning/66731318#66731318
### https://github.com/PyTorchLightning/pytorch-lightning/issues/765
### this is another solution to use the terminal as output console for Pycharm
### https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning
class customTQDMProgressBar(TQDMProgressBar):
    def __init__(self,*args,**kwargs):
        #super(customTQDMProgressBar,self).__init__(**kwargs)
        super().__init__(*args,**kwargs)
        self.status="None"

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["status"]=self.status

        return items

    def init_validation_tqdm(self):
        ### disable validation tqdm instead only use main_progress_bar###
        bar = tqdm(
            disable=True,
        )
        return bar

    def on_validation_epoch_start(self,trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        super().on_validation_epoch_start(trainer,pl_module)
        if not trainer.sanity_checking:
            self.status="Validation"
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer,pl_module)
        self.status = "Done"
        self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer,pl_module)
        self.status = "Training"
        self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    '''
    ### Some testing functions for improving the progressbar ###
    ### But not used to keep the changes more minimal        ###
    def on_train_epoch_start(self, trainer, *_) -> None:
    #    total_train_batches = self.total_train_batches
    #    total_val_batches = self.total_val_batches#
        self.main_progress_bar.reset(convert_inf(self.total_train_batches))
        #self.main_progress_bar.total = convert_inf(self.total_train_batches)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch} - Train     ")

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ### Forcing a line break for having the validation progressbar in a new line   ###
        ### otherwise the train progressbar would be overwritten                       ###
        print(" ")
        ### Update the description of the Progressbar
        #self.main_progress_bar.close()
        self.main_progress_bar.reset(convert_inf(self.total_val_batches))
        #self.main_progress_bar.refresh()
        #self.main_progress_bar.total = convert_inf(self.total_val_batches)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch} - Validation")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ### a little bitt hacky but prevents that the next log or print ###
        ### statement is written into the same line as the progressbar  ###
        super().on_validation_epoch_end(trainer,pl_module)
        
        if not trainer.sanity_checking: print(" ")
    '''

### Callback for measuring the time during train, validation and testing ###
class TimeCallback(Callback):
    def __init__(self):
        self.t_train_start = torch.cuda.Event(enable_timing=True)
        self.t_val_start = torch.cuda.Event(enable_timing=True)
        self.t_test_start = torch.cuda.Event(enable_timing=True)
        self.end=torch.cuda.Event(enable_timing=True)
        self.t_train = []
        self.t_val = []

    def on_train_epoch_start(self, *args, **kwargs):
        self.t_train_start.record()

    def on_validation_epoch_start(self,trainer, *args, **kwargs):

        if not trainer.sanity_checking:

            self.end.record()
            torch.cuda.synchronize()

            train_time = self.t_train_start.elapsed_time(self.end) / 1000
            self.t_train.append(train_time)

            self.log("Time/train_time", train_time, logger=True)
            self.log("Time/mTrainTime", np.mean(self.t_train), logger=True)

        if not trainer.sanity_checking:
            self.t_val_start.record()

    def on_validation_epoch_end(self,trainer, *args, **kwargs):
        if not trainer.sanity_checking:

            self.end.record()
            torch.cuda.synchronize()

            val_time = self.t_val_start.elapsed_time(self.end) / 1000
            self.t_val.append(val_time)

            self.log("Time/validation_time", val_time, logger=True)
            self.log("Time/mValTime", np.mean(self.t_val), logger=True)

    def on_test_epoch_start(self,trainer, *args, **kwargs):

        self.t_test_start.record()

    def on_test_epoch_end(self, *args, **kwargs):


        self.end.record()
        torch.cuda.synchronize()

        test_time = self.t_test_start.elapsed_time(self.end) / 1000

        self.log("Time/test_time", test_time, logger=True)


### Prototype, not used ###
class MS_RestictionCallback(Callback):
    def __init__(self,ms_offset,m_scale_training,epochs):
        if isinstance(ms_offset, int):
            self.ms_offset = ms_offset
        elif isinstance(ms_offset, float):
            self.ms_offset = int(epochs*ms_offset)
        self.m_scale_training=m_scale_training

    def on_train_start(self, trainer,*args, **kwargs):
        if trainer.gpus == 1:
            if self.ms_offset !=0 and self.m_scale_training:
                trainer.model.model.m_scale_training=False
        else:
            if self.ms_offset !=0 and self.m_scale_training:
                trainer.model.module.m_scale_training=False

    def on_train_epoch_start(self, trainer,*args, **kwargs):
        if trainer.gpus==1:
            if trainer.current_epoch==self.ms_offset and self.m_scale_training:
                trainer.model.model.m_scale_training=True
        else:
            if trainer.current_epoch==self.ms_offset and self.m_scale_training:
                trainer.model.module.m_scale_training=True
        #return