import glob
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)
from omegaconf import OmegaConf

import hydra
import argparse
import numpy as np
import torch
import torch.nn.functional as F

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
from src.utils import has_not_empty_attr, get_logger

cv2.setNumThreads(0)

from src.utils import get_logger
from trainers.Instance_Segmentation_Trainer import InstModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.visualization import show_prediction_inst_seg

log = get_logger(__name__)


def predict_img(
    image,
    model,
):
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(model.device))[0]
        pred = [{k: v.detach().cpu() for k, v in pred.items()}]
    return pred


def predict(input_dir, output_dir, overrides_cl):
    hydra.initialize(config_path="../config", version_base="1.1")

    # change working dir to checkpoint dir
    if os.getcwd().endswith("tools"):
        ORG_CWD = os.path.join(os.getcwd(), "..")
    else:
        ORG_CWD = os.getcwd()

    ckpt_dir = None
    for override in overrides_cl:
        if override.startswith("ckpt_dir"):
            ckpt_dir = override.split("=", 1)[1]
            break
    if ckpt_dir is None:
        log.error(
            "ckpt_dir has to be in th config. Run python show_prediction.py ckpt_dir=<some.path>"
        )
        quit()
    os.chdir(ckpt_dir)

    # load overrides from the experiment in the checkpoint dir
    overrides_ckpt = OmegaConf.load(os.path.join("hydra", "overrides.yaml"))
    # compose config by override with overrides_ckpt, afterwards override with overrides_cl
    cfg = hydra.compose(config_name="testing", overrides=overrides_ckpt + overrides_cl)

    # Get the TESTING.OVERRIDES to check if additional parameters should be changed
    if has_not_empty_attr(cfg, "TESTING"):
        if has_not_empty_attr(cfg.TESTING, "OVERRIDES"):
            overrides_test = cfg.TESTING.OVERRIDES
            # Compose config again with including the new overrides
            cfg = hydra.compose(
                config_name="testing",
                overrides=overrides_ckpt + overrides_test + overrides_cl,
            )

    # load the best checkpoint and load the model
    cfg.ORG_CWD = ORG_CWD
    ckpt_file = glob.glob(os.path.join("checkpoints", "best_*"))[0]
    # if hasattr(cfg.MODEL, "PRETRAINED"):
    #    cfg.MODEL.PRETRAINED = False
    model = InstModel.load_from_checkpoint(ckpt_file, model_config=cfg, strict=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    img_files = glob.glob(os.path.join(input_dir, "*.tif*"))

    log.info("{} files found".format(len(img_files)))
    for img_file in img_files:
        log.info("process: {}".format(img_file))
        file_name = img_file.rsplit("/", 1)[1].rsplit(".", 1)[0]
        output_file = os.path.join(output_dir, file_name + ".png")
        if os.path.exists(output_file):
            continue

        img = cv2.imread(os.path.join(img_file), -1)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = np.dstack((img, img, img))
        image = img.astype(np.uint8)

        transform = A.Compose(
            [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
        )
        image_tens = transform(image=image)["image"]
        prediction = predict_img(image_tens, model)

        viz = show_prediction_inst_seg(prediction, img_shape=image_tens.shape[-2:])
        alpha = 0.5
        img_np_fig = cv2.addWeighted(image, 1 - alpha, viz, alpha, 0.0)

        bg_map = np.all(viz == [255, 255, 255], axis=2)
        img_np_fig[bg_map] = image[bg_map]
        bg_map = np.all(viz == [0, 0, 0], axis=2)
        img_np_fig[bg_map] = image[bg_map]

        cv2.imwrite(os.path.join(output_dir, file_name + ".png"), np.array(img_np_fig))


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
    args, overrides = parser.parse_known_args()
    predict(args.input, args.output, overrides)
