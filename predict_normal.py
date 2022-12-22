import glob
import logging

logging.basicConfig(level=logging.INFO)

import os
import hydra
import argparse
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2

cv2.setNumThreads(0)

from src.utils import get_logger

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
            patch = patch.astype(np.float32)
            return patch, patch_indices
        else:
            patch, patch_indices, chunk_id = output
            patch = patch.astype(np.float32)
            return patch, patch_indices, chunk_id

    def __len__(self):
        return len(self.sampler)


def predict_img(
    image,
    model,
    patch_size=(512, 512),
    patch_overlap=(256, 256),
    chunk_size=(600, 600),
    # chunk_size=(4096, 4096),
    # chunk_size=None,
    test_time_augmentation=True,
    no_tqdm=False,
    save_probabilities=False,
):
    spatial_size = image.shape[-2:]
    # Init GridSampler

    # Create an empty prediction passed to the aggregator
    prediction = np.zeros(spatial_size, dtype=np.uint8)

    # Run inference
    with torch.no_grad():
        # for patch, patch_indices in tqdm(loader, disable=no_tqdm):
        image = image.cuda()
        image = image.unsqueeze(0)
        patch_prediction = model(image)

        if test_time_augmentation:
            patch_prediction += torch.flip(model(torch.flip(image.clone(), [2])), [2])
            patch_prediction += torch.flip(model(torch.flip(image.clone(), [3])), [3])
            patch_prediction += torch.flip(model(torch.flip(image.clone(), [2, 3])), [2, 3])
            patch_prediction /= 4

        patch_prediction = patch_prediction.cpu().numpy()
    patch_prediction_argmax = patch_prediction.argmax(1).squeeze(0)
    if not save_probabilities:
        return patch_prediction_argmax
    else:
        patch_prediction_softmax = np.array(
            F.softmax(torch.tensor(patch_prediction.squeeze(0)), -3, _stacklevel=5)
        )

        return patch_prediction_argmax, patch_prediction_softmax


def predict(input_dir, output_dir, overrides, use_tta, no_tqdm=False, save_probabilities=False):
    hydra.initialize(config_path="config", version_base="1.1")
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
        # image = image.transpose(2, 0, 1)
        if not save_probabilities:
            prediction = predict_img(
                image,
                model,
                test_time_augmentation=use_tta,
                no_tqdm=no_tqdm,
                save_probabilities=save_probabilities,
            )
            cv2.imwrite(os.path.join(output_dir, file_name + ".png"), np.array(prediction))
        elif save_probabilities:
            prediction, sm = predict_img(
                image,
                model,
                test_time_augmentation=use_tta,
                no_tqdm=no_tqdm,
                save_probabilities=save_probabilities,
            )
            cv2.imwrite(os.path.join(output_dir, file_name + ".png"), np.array(prediction))
            np.savez(
                os.path.join(
                    output_dir,
                    file_name + ".npz",
                ),
                probabilities=sm,
            )


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
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help="Store Softmax probabilities",
    )

    args, overrides = parser.parse_known_args()
    predict(
        args.input, args.output, overrides, not args.no_tta, args.no_tqdm, args.save_probabilities
    )
