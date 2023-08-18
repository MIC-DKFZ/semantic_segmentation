from os.path import join, split
import os
import json

from datasets.Base_Datasets.base import Base_Dataset
from src.utils.utils import get_logger
from src.utils.dataset_utils import split_by_ID

log = get_logger(__name__)


class CV_Dataset(Base_Dataset):
    def __init__(self, fold=0, num_folds=5, split_ids=None, **kwargs):

        super().__init__(**kwargs)

        self.fold = fold
        self.num_folds = num_folds
        self.split_ids = split_ids

        self.preprocessing_splitting()

        if fold != "all":
            with open(join(self.root, "splits_final.json")) as splits_file:
                splits_final = json.load(splits_file)[self.fold][self.split]

            mask_files_filtered = []
            img_files_filtered = []

            # Just take the image and mask pairs inside the splits_final
            for mask_file, img_file in zip(self.mask_files, self.img_files):
                if any(s in img_file for s in splits_final):
                    mask_files_filtered.append(mask_file)
                    img_files_filtered.append(img_file)

            self.mask_files = mask_files_filtered
            self.img_files = img_files_filtered

        print(
            f"Dataset {self.split} - {len(self.img_files)} Images and {len(self.img_files)} Masks"
        )

    def get_split_ids(self):
        return self.split_ids

    def preprocessing_splitting(self):
        if os.path.exists(join(self.root, "splits_final.json")):
            return
        log.info(f"Split Data into {self.num_folds} Folds")
        print(f"Split Data into {self.num_folds} Folds")

        split_ids = self.get_split_ids()
        files = [split(file)[-1] for file in self.img_files]
        splits = split_by_ID(files, ids=split_ids, num_folds=self.num_folds)

        with open(join(self.root, "splits_final.json"), "w") as file:
            json.dump(splits, file, indent=4)  # [{'train':[],'val':[]}]
