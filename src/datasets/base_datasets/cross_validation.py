from os.path import join, split
import os
import json
from typing import List, Optional

from src.datasets.base_datasets.base import BaseDataset
from src.utils.utils import get_logger
from src.utils.dataset_utils import split_by_ID

log = get_logger(__name__)


class CVDataset(BaseDataset):
    def __init__(
        self, fold: int = 0, num_folds: int = 5, split_ids: Optional[List[str]] = None, **kwargs
    ):
        """

        Parameters
        ----------
        fold: int, optional
            fold which should be used
        num_fold: int, optional
            max number of folds for preprocessing
        split_ids: Optional[List[str]], optional
            ids for defining splits
        kwargs
        """

        # Cross validation related parameters
        self.fold = fold
        self.num_folds = num_folds
        self.split_ids = split_ids

        assert isinstance(fold, int) or fold == "all", "fold is probably not defined in the config"

        super().__init__(**kwargs)

    def preprocessing(self):
        self.preprocessing_cv()
        super().preprocessing()

    # def setup(self):
    #     """
    #     Get Paths to the Image and Mask Files
    #     Only select the files of the current fold and the current split
    #     """
    #     self.preprocessing_cv()
    #     super().setup()

    def setup_files(self):
        """
        Get Paths to the Image and Mask Files
        """
        if self.split == "test" and self.img_folder_test is None:
            log.info("No Test set is defined: Validation set is used instead during testing")
            split = "val"
        else:
            split = self.split

        if split == "test":  # Return test files
            self.img_files: List[str] = self.get_img_files("test")
            self.mask_files: List[str] = self.get_mask_files("test")
        elif self.fold == "all" and split == "val":
            # All files are used for training, so no validation
            self.img_files: List[str] = []
            self.mask_files: List[str] = []
        elif self.fold == "all":  # Return all files
            self.img_files: List[str] = self.get_img_files("train")
            self.mask_files: List[str] = self.get_mask_files("train")
        else:  # Return train or val split
            img_files: List[str] = self.get_img_files("train")
            mask_files: List[str] = self.get_mask_files("train")

            with open(join(self.root, "splits_final.json")) as splits_file:
                splits_final = json.load(splits_file)[self.fold][split]

            mask_files_filtered = []
            img_files_filtered = []

            # Just take the image and mask pairs inside the splits_final
            for mask_file, img_file in zip(mask_files, img_files):
                if any(s in img_file for s in splits_final):
                    mask_files_filtered.append(mask_file)
                    img_files_filtered.append(img_file)

            # Update imgs and masks
            self.mask_files = mask_files_filtered
            self.img_files = img_files_filtered

    def preprocessing_cv(self):
        """
        Preprocessing (only if splits_final.json not already exists):
            Divide split_ids into {self.num_folds} and save in splits_final.json
        """
        if os.path.exists(join(self.root, "splits_final.json")):
            return
        log.info(f"Split Data into {self.num_folds} Folds")

        img_files = self.get_img_files("train")
        split_ids = self.get_split_ids()

        files = [split(file)[-1] for file in img_files]
        splits = split_by_ID(files, ids=split_ids, num_folds=self.num_folds)

        with open(join(self.root, "splits_final.json"), "w") as file:
            json.dump(splits, file, indent=4)  # [{'train':[],'val':[]}]

    def get_split_ids(self) -> List[str]:
        """

        Returns
        -------
        List[str]
            Ids to define the splits
        """
        return self.split_ids
