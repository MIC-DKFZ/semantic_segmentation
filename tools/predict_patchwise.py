import glob
import logging

logging.basicConfig(level=logging.INFO)

import os
import hydra
import argparse
import numpy as np
from samplify.sampler import GridSampler
from samplify.aggregator import Aggregator
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2

cv2.setNumThreads(0)

from src.utils.utils import (
    get_logger,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

log = get_logger(__name__)


class SamplerDataset(Dataset):
    def __init__(self, sampler):
        self.sampler = sampler
        self.is_chunked = self.sampler.chunk_size is not None

    def __getitem__(self, idx):
        output = self.sampler.__getitem__(idx)
        if not self.is_chunked:
            patch, patch_indices = output
            return patch, patch_indices
        else:
            patch, patch_indices, chunk_id = output
            return patch, patch_indices, chunk_id

    def __len__(self):
        return len(self.sampler)


def predict_img(
    image,
    model,
    patch_size=[512, 1024],  # (512, 512),
    patch_overlap=(256, 512),
    chunk_size=None,
    # chunk_size=(700, 700),
    # chunk_size=(4096, 4096),
    test_time_augmentation=True,
    no_tqdm=False,
    num_classes=7,  # 8,
):
    spatial_size = image.shape[-2:]
    # Init GridSampler
    sampler = GridSampler(
        image=image,
        spatial_size=spatial_size,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        chunk_size=chunk_size,
        spatial_first=False,
        mode="sample_edge",
    )
    # Convert sampler into a PyTorch dataset
    loader = SamplerDataset(sampler)
    # Init dataloader
    loader = DataLoader(loader, batch_size=8, num_workers=12, shuffle=False, pin_memory=False)
    # Create an empty prediction passed to the aggregator

    # Init aggregator
    if chunk_size:
        prediction = np.zeros(spatial_size, dtype=np.uint8)
        aggregator = Aggregator(
            sampler=sampler, output=prediction, weights="gaussian", softmax_dim=0
        )
    else:
        prediction = np.zeros((num_classes, *spatial_size), dtype=np.float32)
        aggregator = Aggregator(
            sampler=sampler, output=prediction, weights="gaussian", softmax_dim=None
        )

    # Run inference
    with torch.no_grad():
        # for patch, patch_indices, chunk_id in tqdm(loader, disable=no_tqdm):
        for _patch in tqdm(loader, disable=no_tqdm):
            if chunk_size:
                patch, patch_indices, chunk_id = _patch
            else:
                patch, patch_indices = _patch
            patch = patch.cuda()
            patch_prediction = model(patch)

            if test_time_augmentation:
                patch_prediction += torch.flip(model(torch.flip(patch.clone(), [2])), [2])
                patch_prediction += torch.flip(model(torch.flip(patch.clone(), [3])), [3])
                patch_prediction += torch.flip(model(torch.flip(patch.clone(), [2, 3])), [2, 3])
                patch_prediction /= 4

            patch_prediction = patch_prediction.cpu().numpy()
            # print("PS", patch_prediction.shape)
            patch_indices = patch_indices.cpu().numpy()
            if chunk_size:
                chunk_id = chunk_id.cpu().numpy()
                for i in range(len(patch_prediction)):
                    aggregator.append(patch_prediction[i], patch_indices[i], chunk_id[i])
            else:
                for i in range(len(patch_prediction)):
                    aggregator.append(patch_prediction[i], patch_indices[i])

    # Finalize aggregation
    prediction_sm = aggregator.get_output()
    if chunk_size is None:
        prediction = prediction_sm.argmax(0)
    return prediction, prediction_sm


def predict(input_dir, output_dir, overrides, use_tta, no_tqdm=False):
    hydra.initialize(config_path="../config", version_base="1.1")
    cfg = hydra.compose(config_name="baseline", overrides=overrides)
    model = hydra.utils.instantiate(cfg.model)
    model.eval().to("cuda")

    img_files = glob.glob(os.path.join(input_dir, "*.png"))

    log.info("{} files found".format(len(img_files)))
    for img_file in img_files:
        log.info("process: {}".format(img_file))
        file_name = img_file.rsplit("/", 1)[1].rsplit(".", 1)[0]
        output_file = os.path.join(output_dir, file_name + ".png")
        if os.path.exists(output_file):
            continue

        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = A.Compose(
            [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
        )
        image = transform(image=image)["image"]

        prediction = predict_img(image, model, test_time_augmentation=use_tta, no_tqdm=no_tqdm)
        # log.info("classes", np.unique(prediction))
        cv2.imwrite(os.path.join(output_dir, file_name + ".png"), np.array(prediction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Input",
        default="/home/l727r/Desktop/Datasets/Diadem_example/imgs",
    )
    parser.add_argument(
        "-o", "--output", help="Output", default="/home/l727r/Desktop/Datasets/Diadem_example/preds"
    )
    parser.add_argument(
        "--no_tqdm",
        action="store_true",
        help="No TQDM",
    )
    parser.add_argument(
        "--no_tta",
        action="store_true",
        help="No TQDM",
    )

    args, overrides = parser.parse_known_args()
    predict(args.input, args.output, overrides, not args.no_tta, args.no_tqdm)
