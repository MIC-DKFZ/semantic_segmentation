import random
import numpy as np

from io import BytesIO
from PIL import Image
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import staintools
from utils.staintools.stain_augmentor import StainAugmentor
from utils.staintools.stain_normalizer import StainNormalizer

light_augmentor = StainAugmentor(method="vahadane", sigma1=0.2, sigma2=0.2)
strong_augmentor = StainAugmentor(method="vahadane", sigma1=-0.8, sigma2=-0.8)
vahadane_normalizer = StainNormalizer(method="vahadane")
# ReferenceFileForNormalization = staintools.read_image(
#    "/dkfz/cluster/gpu/data/OE0441/l727r/AGGC2022/Patch_X_10984_Y_73587.png"
# )


class LightAugmentor(ImageOnlyTransform):
    def apply(self, img, **params):
        return VahadaneLightAugmentor(img)


class StrongAugmentor(ImageOnlyTransform):
    def apply(self, img, **params):
        return VahadaneStrongAugmentor(img)


class VahadaneNormalizer(ImageOnlyTransform):
    def apply(self, img, **params):
        return Vahadane_Normalizer(img)


def Vahadane_Normalizer(image):
    vahadane_normalizer.fit(ReferenceFileForNormalization)
    return vahadane_normalizer.transform(image)


def VahadaneLightAugmentor(image):
    light_augmentor.fit(image)
    return light_augmentor.pop()


def VahadaneStrongAugmentor(image):
    strong_augmentor.fit(image)
    return strong_augmentor.pop()


def RandomJPEGcompression(image):
    quality = random.randrange(1, 95)
    outputIoStream = BytesIO()
    # image = np.array(image, dtype = np.uint8)
    image = Image.fromarray(image)
    image.save(outputIoStream, "JPEG", quality=quality)
    outputIoStream.seek(0)
    img = Image.open(outputIoStream)
    return np.array(img, dtype=np.uint8)


class JPEGcompressor(ImageOnlyTransform):
    def apply(self, img, **params):
        return RandomJPEGcompression(img)


def randAugment_AGGC(N, M, p=0.5):
    # p = 1
    # https://towardsdatascience.com/augmentation-methods-using-albumentations-and-pytorch-35cd135382f8
    # https://openreview.net/pdf?id=JrBfXaoxbA2

    # Magnitude(M) search space
    contrast = [np.linspace(-0.8, -0.1, 10), np.linspace(0.1, 2, 10)]
    brightness = np.linspace(0.1, 0.7, 15)
    sharpness = np.linspace(0, 0.5, 15)
    sharpness = np.linspace(0.1, 0.9, 10)
    rotation = np.linspace(0, 90, 15)
    translate_x = np.linspace(-30, 30, 15, dtype=int)
    translate_y = np.linspace(-30, 30, 15, dtype=int)
    shear_x = np.linspace(0, 10, 15)
    shear_y = np.linspace(0, 10, 15)
    HSV_shift = np.linspace(0, 0.5, 15)
    HED_shift = np.linspace(-0.5, 0.5, 15)
    color = np.linspace(0, 1, 15)

    # Augmentations
    color_Aug = [
        # HSV shift
        A.ColorJitter(
            brightness=0,
            contrast=0,
            saturation=abs(HSV_shift[M]),
            hue=abs(HSV_shift[M]),
            p=p,
        ),
        # Contrast
        # A.RandomBrightnessContrast(contrast_limit=contrast[M], brightness_limit=0, p=p),
        A.RandomBrightnessContrast(
            contrast_limit=(contrast[0][M], contrast[1][M]), brightness_limit=0, p=p
        ),
        # Brightness
        A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=brightness[M], p=p),
        # A.RandomBrightness(limit=brightness[M], p=p),
        # Sharpness
        A.UnsharpMask(p=p),
        # A.Sharpen(alpha=[1, 1], lightness=sharpness[M], p=p),
        # A.Sharpen(alpha=sharpness[M], lightness=sharpness[M], p=p),
        ## Rotate
        A.Rotate(limit=rotation[M], p=p),
        # Translate x and y
        # A.Affine(translate_px={"x": (-abs(translate_x[M]), abs(translate_x[M])), "y": 0}, p=p),
        # A.Affine(translate_px={"x": 0, "y": (-abs(translate_y[M]), abs(translate_y[M]))}, p=p),
        # Shear x and y
        A.Affine(
            shear={"x": (-abs(shear_x[M]), abs(shear_x[M])), "y": 0},
            p=p,
            # fit_output=True,
            # keep_ratio=True,
            mode=2,
        ),
        A.Affine(
            shear={"x": 0, "y": (abs(shear_y[M]), abs(shear_y[M]))},
            p=p,
            # fit_output=True,
            # keep_ratio=True,
            mode=2,
        ),
        # Noise/Blur
        A.AdvancedBlur(p=p),
    ]

    # Sampling from the Transformation search space
    # color_Aug = [A.AdvancedBlur(p=1), A.UnsharpMask(p=1)]
    # ops = np.random.choice(color_Aug, N)
    # transforms = A.Compose(color_Aug)
    transforms = A.SomeOf(color_Aug, n=N)

    return transforms  # , ops


def randAugment_AGGC_equ(N, M, p=0.5):
    # p = 1
    # https://towardsdatascience.com/augmentation-methods-using-albumentations-and-pytorch-35cd135382f8
    # https://openreview.net/pdf?id=JrBfXaoxbA2
    # Magnitude(M) search space
    contrast = [np.linspace(-0.8, -0.1, 10), np.linspace(0.1, 2, 10)]
    brightness = np.linspace(0.1, 0.7, 15)
    rotation = np.linspace(0, 90, 15)
    shear_x = np.linspace(0, 10, 15)
    shear_y = np.linspace(0, 10, 15)
    HSV_shift = np.linspace(0, 0.5, 15)
    HED_shift = np.linspace(-0.5, 0.5, 15)

    color_Aug = [
        # HSV shift
        A.ColorJitter(
            brightness=0,
            contrast=0,
            saturation=abs(HSV_shift[M]),
            hue=abs(HSV_shift[M]),
            p=p,
        ),
        # Contrast
        # A.RandomBrightnessContrast(contrast_limit=contrast[M], brightness_limit=0, p=p),
        A.RandomBrightnessContrast(
            contrast_limit=(contrast[0][M], contrast[1][M]), brightness_limit=0, p=p
        ),
        # Brightness
        A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=brightness[M], p=p),
        # Sharpness
        A.UnsharpMask(p=p),
        ## Rotate
        A.Rotate(limit=rotation[M], p=p),
        # Shear x and y
        A.Affine(
            shear={"x": (-abs(shear_x[M]), abs(shear_x[M])), "y": 0},
            p=p,
            mode=2,
        ),
        A.Affine(
            shear={"x": 0, "y": (abs(shear_y[M]), abs(shear_y[M]))},
            p=0,
            mode=2,
        ),
        A.AdvancedBlur(p=p),
        JPEGcompressor(p=p),
        # Equalize
        A.Equalize(p=p),
    ]
    transforms = A.SomeOf(color_Aug, n=N)

    return transforms


def randAugment_AGGC_light(N, M, p=0.5):
    # p = 1
    # https://towardsdatascience.com/augmentation-methods-using-albumentations-and-pytorch-35cd135382f8
    # https://openreview.net/pdf?id=JrBfXaoxbA2
    # Magnitude(M) search space
    contrast = [np.linspace(-0.8, -0.1, 10), np.linspace(0.1, 2, 10)]
    brightness = np.linspace(0.1, 0.7, 15)
    rotation = np.linspace(0, 90, 15)
    shear_x = np.linspace(0, 10, 15)
    shear_y = np.linspace(0, 10, 15)
    HSV_shift = np.linspace(0, 0.5, 15)
    HED_shift = np.linspace(-0.5, 0.5, 15)

    color_Aug = [
        # HSV shift
        A.ColorJitter(
            brightness=0,
            contrast=0,
            saturation=abs(HSV_shift[M]),
            hue=abs(HSV_shift[M]),
            p=p,
        ),
        # Contrast
        # A.RandomBrightnessContrast(contrast_limit=contrast[M], brightness_limit=0, p=p),
        A.RandomBrightnessContrast(
            contrast_limit=(contrast[0][M], contrast[1][M]), brightness_limit=0, p=p
        ),
        # Brightness
        A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=brightness[M], p=p),
        # Sharpness
        A.UnsharpMask(p=p),
        ## Rotate
        A.Rotate(limit=rotation[M], p=p),
        # Shear x and y
        A.Affine(
            shear={"x": (-abs(shear_x[M]), abs(shear_x[M])), "y": 0},
            p=p,
            mode=2,
        ),
        A.Affine(
            shear={"x": 0, "y": (abs(shear_y[M]), abs(shear_y[M]))},
            p=0,
            mode=2,
        ),
        A.AdvancedBlur(p=p),
        JPEGcompressor(p=p),
        # HED Augmentations
        LightAugmentor(p=p),
        # StrongAugmentor(p=p),
        # Equalize
        # A.Equalize(p=p),
    ]
    transforms = A.SomeOf(color_Aug, n=N)

    return transforms


def randAugment_AGGC_strong(N, M, p=0.5):
    # p = 1
    # https://towardsdatascience.com/augmentation-methods-using-albumentations-and-pytorch-35cd135382f8
    # https://openreview.net/pdf?id=JrBfXaoxbA2
    # Magnitude(M) search space
    contrast = [np.linspace(-0.8, -0.1, 10), np.linspace(0.1, 2, 10)]
    brightness = np.linspace(0.1, 0.7, 15)
    rotation = np.linspace(0, 90, 15)
    shear_x = np.linspace(0, 10, 15)
    shear_y = np.linspace(0, 10, 15)
    HSV_shift = np.linspace(0, 0.5, 15)
    HED_shift = np.linspace(-0.5, 0.5, 15)

    color_Aug = [
        # HSV shift
        A.ColorJitter(
            brightness=0,
            contrast=0,
            saturation=abs(HSV_shift[M]),
            hue=abs(HSV_shift[M]),
            p=p,
        ),
        # Contrast
        # A.RandomBrightnessContrast(contrast_limit=contrast[M], brightness_limit=0, p=p),
        A.RandomBrightnessContrast(
            contrast_limit=(contrast[0][M], contrast[1][M]), brightness_limit=0, p=p
        ),
        # Brightness
        A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=brightness[M], p=p),
        # Sharpness
        A.UnsharpMask(p=p),
        ## Rotate
        A.Rotate(limit=rotation[M], p=p),
        # Shear x and y
        A.Affine(
            shear={"x": (-abs(shear_x[M]), abs(shear_x[M])), "y": 0},
            p=p,
            mode=2,
        ),
        A.Affine(
            shear={"x": 0, "y": (abs(shear_y[M]), abs(shear_y[M]))},
            p=p,
            mode=2,
        ),
        A.AdvancedBlur(p=p),
        JPEGcompressor(p=p),
        # HED Augmentations
        # LightAugmentor(p=p),
        StrongAugmentor(p=p),
        # Equalize
        # A.Equalize(p=p),
    ]
    transforms = A.SomeOf(color_Aug, n=N)

    return transforms
