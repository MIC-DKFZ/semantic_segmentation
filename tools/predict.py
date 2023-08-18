import glob
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)


import hydra
import argparse
import numpy as np
import torch
import torch.nn.functional as F

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2

cv2.setNumThreads(0)

from src.utils.utils import get_logger

import albumentations as A
from albumentations.pytorch import ToTensorV2

log = get_logger(__name__)


def predict_img(
    image,
    model,
    test_time_augmentation=True,
    save_probabilities=False,
):
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


def predict(input_dir, output_dir, overrides, use_tta, save_probabilities=False):
    hydra.initialize(config_path="../config", version_base="1.3")
    cfg = hydra.compose(config_name="training", overrides=overrides)
    model = hydra.utils.instantiate(cfg.model)
    model.eval().to("cuda")

    os.makedirs(output_dir, exist_ok=True)

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
                save_probabilities=save_probabilities,
            )
            cv2.imwrite(os.path.join(output_dir, file_name + ".png"), np.array(prediction))
        elif save_probabilities:
            prediction, sm = predict_img(
                image,
                model,
                test_time_augmentation=use_tta,
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
    predict(args.input, args.output, overrides, not args.no_tta, args.save_probabilities)
