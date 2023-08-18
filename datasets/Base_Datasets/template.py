import torch


class Base_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        train_imgs = ...
        train_labels = ...
        val_imgs = ...
        val_labels = ...
        test_imgs = ...
        val_imgs = ...

        print("info")

    def load_img(self):
        pass

    def load_mask(self):
        pass

    def load_data(self):
        pass

    def apply_transforms(self):
        pass

    def __getitem__(self, idx: int) -> tuple:
        pass

    def __len__(self) -> int:
        pass
