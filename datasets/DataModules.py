import numpy as np
import albumentations as A

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from datasets.Cityscape import Cityscape_dataset
from datasets.Cityscape_Coarse import Cityscape_coarse_dataset
from config.utils import hasNotEmptyAttr
import hydra
from omegaconf import OmegaConf
import logging
log = logging.getLogger(__name__)

def collate_fn(batch):

    img=[x[0] for x in batch]
    img=torch.stack((img), dim=0)

    labels = [x[1] for x in batch]
    labels = torch.stack((labels), dim=0)

    org = [x[2] for x in batch]

    return (img,labels,org)


class BaseDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size,val_batch_size, num_workers ,augmentations, train_size):
        super().__init__()
        self.num_workers = num_workers
        self.augmentations = augmentations
        self.base_size = train_size
        self.batch_size = batch_size
        self.val_batch_size=val_batch_size

        self.dataset=dataset

    def setup(self, stage= None):

        transforms_train = self.get_augmentations_from_config(self.augmentations.TRAIN)
        transforms_val = self.get_augmentations_from_config(self.augmentations.TEST)
        self.transforms_train=transforms_train

        if stage in (None, "fit"):
            self.CS_train = hydra.utils.instantiate(self.dataset, split="train", transforms=transforms_train)
        if stage in (None,"fit","validate"):
            self.CS_val = hydra.utils.instantiate(self.dataset, split="val", transforms=transforms_val)
        if stage in (None, "test"):
            self.CS_test = hydra.utils.instantiate(self.dataset, split="test", transforms=transforms_val)

    def max_steps(self):
        # dataset size
        # batch size
        # num gpus
        # epochs
        # acc?
        steps_per_epoch = self.base_size // self.batch_size
        steps_per_gpu = int(np.ceil(steps_per_epoch / self.trainer.gpus))
        acc_steps_per_gpu = int(np.ceil(steps_per_gpu / self.trainer.accumulate_grad_batches))
        max_steps = (self.trainer.max_epochs * acc_steps_per_gpu)

        log.info("MAX STEPS: %s", max_steps)
        return max_steps

    def get_augmentations_from_config(self,augmentations):
        if hasattr(augmentations, "FROM_DICT"):
            if augmentations.FROM_DICT is not None:
                return A.from_dict(OmegaConf.to_container(augmentations.FROM_DICT))
        transforms = list(augmentations.keys())
        trans = []
        for transform in transforms:
            parameters = getattr(augmentations, transform)
            if parameters == None: parameters = {}
            try:
                # try to load the functions from ALbumentations(A)
                func = getattr(A, transform)
                trans.append(func(**parameters))
            except:
                try:
                    # exeption for ToTensorV2 function which is in A.pytorch
                    func = getattr(A.pytorch, transform)
                    trans.append(func(**parameters))
                except:
                    print("No Operation Found", transform)
        return A.Compose(trans)

    def train_dataloader(self):
        return DataLoader(self.CS_train,shuffle=True, pin_memory=True,batch_size=self.batch_size,num_workers=self.num_workers,drop_last=True,persistent_workers=True)#,collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.CS_val, pin_memory=True,batch_size=self.val_batch_size,num_workers=self.num_workers,persistent_workers=True)#,collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.CS_test,pin_memory=True, batch_size=self.val_batch_size,num_workers=self.num_workers,persistent_workers=True)

class PascalModule(BaseDataModule):
    def val_dataloader(self):
        return DataLoader(self.CS_val, pin_memory=True,batch_size=self.val_batch_size,num_workers=self.num_workers,persistent_workers=True,collate_fn=collate_fn)

class Cityscape(BaseDataModule):
    def __init__(self,config):
        super().__init__(config)
        self.dataset=Cityscape_dataset

class Cityscape_Coarse(BaseDataModule):
    def __init__(self,dataset_extra,coarse_size,start_portion,max_coarse_epoch, **kwargs):
        super().__init__(**kwargs)

        self.coarse_size = coarse_size
        self.start_portion = start_portion
        self.max_coarse_epoch = max_coarse_epoch
        self.dataset_extra=dataset_extra
        #if hasattr(config.DATASET, "COARSE_DATA"):
        #    if hasattr(config.DATASET.COARSE_DATA, "START_PORTION"):
        #        self.start_portion = config.DATASET.COARSE_DATA.START_PORTION
        #    if hasattr(config.DATASET.COARSE_DATA, "MAX_COARSE_EPOCH"):
        #            self.max_coarse_epoch = config.DATASET.COARSE_DATA.MAX_COARSE_EPOCH

        self.policy=lambda current_epoch: self.start_portion * np.maximum(1 - current_epoch / (self.max_coarse_epoch - 1), 0)

    def max_steps(self):

        if self.max_coarse_epoch == None:
            self.max_coarse_epoch = self.trainer.max_epochs

        max_epoch = self.trainer.max_epochs
        gpus = self.trainer.gpus
        acc_grad_batches = self.trainer.accumulate_grad_batches

        epochs = np.array(range(max_epoch))
        data_size = (self.policy(epochs) * self.coarse_size).astype(int) + self.base_size

        size_dataloader = data_size // self.batch_size
        steps_per_gpu = (np.ceil(size_dataloader / gpus)).astype(int)
        acc_steps_per_gpu = (np.ceil(steps_per_gpu / acc_grad_batches)).astype(int)
        max_steps = sum(acc_steps_per_gpu)
        log.info("MAX STEPS: %s", max_steps)
        return max_steps

    def train_dataloader(self):
        coarse_portion=self.policy(self.trainer.current_epoch)
        #self.CS_train = Cityscape_coarse_dataset(root=self.root, split="train", num_classes=self.num_classes,
        #                                         transforms=self.transforms_train, coarse_portion=coarse_portion)
        self.CS_train = hydra.utils.instantiate(self.dataset_extra, split="train", transforms=self.transforms_train, coarse_portion=coarse_portion)
        return DataLoader(self.CS_train, shuffle=True, pin_memory=True, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=True)

