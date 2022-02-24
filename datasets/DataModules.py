import hydra
from omegaconf import OmegaConf
import numpy as np

import albumentations as A
import albumentations.pytorch

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from utils.utils import hasNotEmptyAttr
from utils.utils import get_logger

log = get_logger(__name__)

class BaseDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size,val_batch_size, num_workers ,augmentations):
        super().__init__()

        ### PARAMETERS FOR DATALOADER
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size=val_batch_size
        ### DATA AUGMENTATIONS FOR TRAIN,VAL AND TEST
        self.augmentations = augmentations
        ### DATASET WHICH IS DEFINED IN THE CONFIG
        self.dataset=dataset

    def setup(self, stage= None):

        ### GET THE AUGMENTATIONS FOR TRAIN,VALIDATION AND TEST SET ###
        transforms_train = self.get_augmentations_from_config(self.augmentations.TRAIN)[0]
        transforms_val = self.get_augmentations_from_config(self.augmentations.VALIDATION)[0]
        transforms_test = self.get_augmentations_from_config(self.augmentations.TEST)[0]

        ### DEFINE THE DATASETS WHICH ARE DEFINED IN THE CONFIG ###
        ### ADDITIONAL ARGUMENTS ARE THE SPLIT AND THE AUGMENTATIONS ###
        if stage in (None, "fit"):
            self.DS_train = hydra.utils.instantiate(self.dataset, split="train", transforms=transforms_train)
        if stage in (None,"fit","validate"):
            self.DS_val = hydra.utils.instantiate(self.dataset, split="val", transforms=transforms_val)
        if stage in (None, "test"):
            self.DS_test = hydra.utils.instantiate(self.dataset, split="test", transforms=transforms_test)

    def max_steps(self):
        ### COMPUTING THE MAXIMAL NUMBER OF STEPS FOR TRAINING
        base_size = len(self.DS_train)
        steps_per_epoch = base_size // self.batch_size
        steps_per_gpu = int(np.ceil(steps_per_epoch / self.trainer.num_gpus))
        acc_steps_per_gpu = int(np.ceil(steps_per_gpu / self.trainer.accumulate_grad_batches))
        max_steps = (self.trainer.max_epochs * acc_steps_per_gpu)

        log.info("MAX STEPS: %s", max_steps)
        return max_steps

    def get_augmentations_from_config(self,augmentations):
        ### RECURSIVE FUNCTION FOR GETTING THE ALBUMENTATIONS DATA PIPELINE WHICH IS DEFINED IN THE CONFIG ###
        if hasNotEmptyAttr(augmentations, "FROM_DICT"):
            return [A.from_dict(OmegaConf.to_container(augmentations.FROM_DICT))]

        trans = []
        for augmentation in augmentations:

            transforms = list(augmentation.keys())

            for transform in transforms:
                #print("TF",transform)
                parameters = getattr(augmentation, transform)
                if parameters == None: parameters = {}

                if "transforms" in list(parameters.keys()):
                    transforms=self.get_augmentations_from_config(parameters.transforms)
                    del parameters["transforms"]
                    func = getattr(A, transform)
                    trans.append(func(transforms=transforms,**parameters))
                else:
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
                            log.info("No Operation Found: %s", transform)
        return trans

    def train_dataloader(self):
        return DataLoader(self.DS_train,shuffle=True, pin_memory=True,batch_size=self.batch_size,num_workers=self.num_workers,drop_last=True,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.DS_val, pin_memory=True,batch_size=self.val_batch_size,num_workers=self.num_workers,persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.DS_test,pin_memory=True, batch_size=self.val_batch_size,num_workers=self.num_workers,persistent_workers=True)

