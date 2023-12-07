from typing import List, Union, Tuple, Any
from abc import ABC, abstractmethod
import glob

from numpy import ndarray
from os.path import join
from torch import Tensor
import albumentations as A
import torch.nn as nn


class BaseLoader(ABC):
    def __init__(self, prefix: str = "", postfix: str = "", dtype: str = ".png"):
        self.prefix = prefix
        self.postfix = postfix
        self.dtype = dtype

    def get_files(self, path: str) -> List[str]:
        files = glob.glob(join(path, f"{self.prefix}*{self.postfix}{self.dtype}"))
        return list(sorted(files))

    @abstractmethod
    def load_file(self, file: str) -> ndarray:
        pass


class BaseLabelHandler(BaseLoader):
    def __init__(self, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    @abstractmethod
    def apply_transforms(
        self, img: ndarray, mask: ndarray, transforms: A.transforms, *args, **kwargs
    ) -> Tuple[Union[ndarray, Any, Tensor], Union[ndarray, Any, Tensor]]:
        pass

    """
    Needed for Sampling 
    """

    @abstractmethod
    def get_class_ids(self, mask: ndarray) -> Union[ndarray, List[int]]:
        pass

    @abstractmethod
    def get_class_locations(
        self, mask: ndarray, class_id: int
    ) -> Union[Tuple[ndarray, ndarray], Tuple[List[int], List[int]]]:
        pass

    """
    Needed Prediction Writer
    """

    @abstractmethod
    def to_cpu(self, pred: Tensor) -> Tensor:
        pass

    @abstractmethod
    def save_prediction(self, logits: Tensor, file: str) -> None:
        pass

    @abstractmethod
    def save_probabilities(self, logits: Tensor, file: str) -> None:
        pass

    @abstractmethod
    def save_visualization(self, logits: Tensor, file: str) -> None:
        pass

    """
    Needed for Visualization
    """

    @abstractmethod
    def predict_img(self, img: Tensor, model: nn.Module) -> Tensor:
        pass

    @abstractmethod
    def viz_mask(self, mask: Tensor, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def viz_prediction(self, logits: Tensor, *args, **kwargs) -> None:
        pass
