import os

import torch
import numpy as np
import cv2
import albumentations as A
from src.utils.utils import get_logger
from PIL import Image
import json

log = get_logger(__name__)


class DIADEM_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms, fold=0):
        # get your data for the corresponding split
        if split == "test":
            split = "val"

        with open(os.path.join(root, "splits_final.json")) as f:
            splits = json.load(f)[fold][split]

        self.imgs = [os.path.join(root, "imagesTr", name + "_0000.png") for name in splits]
        self.masks = [os.path.join(root, "labelsTr", name + ".png") for name in splits]

        # save the transformations
        self.transforms = transforms

    def __getitem__(self, idx):
        # reading images and masks as numpy arrays
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads images in BGR order

        mask = cv2.imread(self.masks[idx], -1)

        # thats how you apply Albumentations transformations
        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    import torchvision.transforms.functional as F

    color_Aug = [
        # A.RandomBrightnessContrast(contrast_limit=(1, 1), brightness_limit=0, p=1),
        # ToTensorV2(),
    ]

    # Sampling from the Transformation search space
    # color_Aug = [A.AdvancedBlur(p=1), A.UnsharpMask(p=1)]
    # ops = np.random.choice(color_Aug, N)
    transforms = A.Compose([A.SmallestMaxSize(max_size=512)])
    root = "/home/l727r/Desktop/DIADEM/Dataset101_DIADEM"
    dataset = DIADEM_dataset(root, "train", transforms)
    img, mask = dataset[0]
    # img = img[:, 0:-300, :]
    ###################
    fig_a = img.copy()
    fig_t = img.copy()
    cv2.putText(
        fig_a,
        "A",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        fig_t,
        "T",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    ###################
    bins = 10

    min_a = 10
    max_a = 0
    range_a = np.linspace(min_a, max_a, bins)

    min_t = 0
    max_t = 1.9
    range_t = np.linspace(min_t, max_t, bins)
    ###################

    for val in range_a:
        transforms = A.Compose(
            [A.Blur(blur_limit=3, p=1)]
            # [A.Sharpen(lightness=(val, val), alpha=(1, 1), p=1)]
            # [A.UnsharpMask(sigma_limit=val, alpha=(0.2, np.clip(val, 0.2, 1)), p=1)]
            # [A.ColorJitter(brightness=(val, val), contrast=0, saturation=0, hue=0, p=1)]
            # [A.Solarize(threshold=(val, 255), p=1)]
            # [c]
        )
        # print(A.Sharpen(lightness=(val, val), p=1).get_params())

        im_a = transforms(image=img.copy())["image"]
        print(im_a == img)
        cv2.putText(
            im_a,
            str(val)[0:8],
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        fig_a = np.concatenate((fig_a, im_a), 1)

    for val in range_t:
        im_t = F.adjust_sharpness(Image.fromarray(img.copy()), val)
        # im_t = F.adjust_brightness(Image.fromarray(img.copy()), val)
        # im = F.adjust_brightness(Image.fromarray(img), val)
        # im = F.adjust_contrast(Image.fromarray(img), val)
        # im = F.adjust_sharpness(Image.fromarray(img), val)
        # im = F.adjust_saturation(Image.fromarray(img), val)
        # im_t = F.solarize(Image.fromarray(img.copy()), int(val))
        # im_t = F.rotate(
        #    Image.fromarray(img.copy()),
        #    val,
        #    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        #    fill=None,
        # )
        # im = F.posterize(Image.fromarray(img), int(val))
        # im = F.invert(Image.fromarray(img))
        im_t = np.array(im_t)
        cv2.putText(
            im_t,
            str(val)[0:8],
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        fig_t = np.concatenate((fig_t, im_t), 1)
    # print(range)
    # print(fig.dtype)
    # fig.resize(target_shape, resample=Image.NEAREST)
    fig_a = Image.fromarray(fig_a)
    fig_t = Image.fromarray(fig_t)
    # w, h = fig.size
    fig_a.show()
    fig_t.show()
    # im.show()
    # plt.imshow(transforms.ToPILImage()(img).convert("RGB"))
