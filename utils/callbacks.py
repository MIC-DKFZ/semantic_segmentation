import time
import numpy as np
import torch

from pytorch_lightning.callbacks import Callback, ModelCheckpoint

class customModelCheckpoint(ModelCheckpoint):
    def __init__(self,**kwargs):
        super(customModelCheckpoint,self,).__init__(**kwargs)
        self.CHECKPOINT_NAME_LAST="last_{epoch}"


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