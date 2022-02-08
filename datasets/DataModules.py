import logging
import hydra
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import albumentations as A
import albumentations.pytorch
import numpy as np

from utils.utils import hasNotEmptyAttr

log = logging.getLogger(__name__)

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
        transforms_train = self.get_augmentations_from_config(self.augmentations.TRAIN)[0]
        transforms_val = self.get_augmentations_from_config(self.augmentations.VALIDATION)[0]
        transforms_test = self.get_augmentations_from_config(self.augmentations.TEST)[0]

        if stage in (None, "fit"):
            self.DS_train = hydra.utils.instantiate(self.dataset, split="train", transforms=transforms_train)
        if stage in (None,"fit","validate"):
            self.DS_val = hydra.utils.instantiate(self.dataset, split="val", transforms=transforms_val)
        if stage in (None, "test"):
            self.DS_test = hydra.utils.instantiate(self.dataset, split="test", transforms=transforms_test)

    def max_steps(self):

        steps_per_epoch = self.base_size // self.batch_size
        steps_per_gpu = int(np.ceil(steps_per_epoch / self.trainer.num_gpus))
        acc_steps_per_gpu = int(np.ceil(steps_per_gpu / self.trainer.accumulate_grad_batches))
        max_steps = (self.trainer.max_epochs * acc_steps_per_gpu)

        log.info("MAX STEPS: %s", max_steps)
        return max_steps

    def get_augmentations_from_config(self,augmentations):

        if hasNotEmptyAttr(augmentations, "FROM_DICT"):
            return A.from_dict(OmegaConf.to_container(augmentations.FROM_DICT))

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
                            print("No Operation Found", transform)
        return trans

    def train_dataloader(self):
        return DataLoader(self.DS_train,shuffle=True, pin_memory=True,batch_size=self.batch_size,num_workers=self.num_workers,drop_last=True,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.DS_val, pin_memory=True,batch_size=self.val_batch_size,num_workers=self.num_workers,persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.DS_test,pin_memory=True, batch_size=self.val_batch_size,num_workers=self.num_workers,persistent_workers=True)

