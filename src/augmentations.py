import random
import numpy as np

from io import BytesIO
from PIL import Image
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


def RandAugment(N: int, M: int, p: float = 0.5, bins: int = 10, mode: int = 0) -> object:

    """
    https://towardsdatascience.com/augmentation-methods-using-albumentations-and-pytorch-35cd135382f8
    # https://openreview.net/pdf?id=JrBfXaoxbA2
    # https://arxiv.org/pdf/1805.09501.pdf
    Parameters
    ----------
    N: int
        Number of Operations
    M: int
        Magnitude, intensity of the operations
    p: float, optional
        Probability for each operation to be applied
    bins: int, optional
        Number of Bins, max value for M
    mode: int, optional
        Border Mode for geometric transforms
        BORDER_CONSTANT    = 0,
        BORDER_REPLICATE   = 1,
        BORDER_REFLECT     = 2,
        BORDER_WRAP        = 3,
        BORDER_REFLECT_101 = 4,
        BORDER_TRANSPARENT = 5,
        BORDER_REFLECT101  = BORDER_REFLECT_101,
        BORDER_DEFAULT     = BORDER_REFLECT_101,
        BORDER_ISOLATED    = 16,
    """
    """
    Color Transformations
    """
    # Contrast: 1=original, 0=gray, Range [max(0, 1 - contrast), 1 + contrast]
    contrast = np.linspace(0.0, 0.9, bins)
    # Brightness: 1=original, 0=black, Range [max(0, 1 - brightness), 1 + brightness]
    brightness = np.linspace(0.0, 0.9, bins)
    # Color: 1=original, 0=black-white image, Range [max(0, 1 - color), 1 + color]
    color = np.linspace(0.0, 0.9, bins)
    # Solariize: 255=original, 0=inverted, Range [solarize,255]
    solarize = np.linspace(255, 0, bins)
    # Posterize: Range [8,4]
    posterize = (8 - (np.arange(bins) / ((bins - 1) / 4)).round()).astype(int)
    # Sharpen
    sharpen = np.linspace(0.0, 2.0, bins)

    col_transforms = [
        # Identity
        A.NoOp(p=p),
        # Contrast - equivalent to F.adjust_contrast(Image.fromarray(img), val)
        A.ColorJitter(brightness=0, contrast=contrast[M], saturation=0, hue=0, p=p),
        # Brightness - equivalent to F.adjust_brightness(Image.fromarray(img), val)
        A.ColorJitter(brightness=brightness[M], contrast=0, saturation=0, hue=0, p=p),
        # Color (~Saturation) - equivalent to F.adjust_saturation(Image.fromarray(img), val)
        A.ColorJitter(brightness=0, contrast=0, saturation=color[M], hue=0, p=p),
        # Solarize - eqilvalent to F.solarize(Image.fromarray(img), int(val))
        A.Solarize(threshold=(solarize[M], 255), p=p),
        # Posterize - F.posterize(Image.fromarray(img), int(val))
        A.Posterize(num_bits=int(posterize[M]), p=p),
        # Equalize - equivalent to F.equalize(Image.fromarray(img))
        A.Equalize(mode="pil", p=p),
        # Invert - equalize to F.invert(Image.fromarray(img))
        A.InvertImg(p=p),
        # Sharpen - no equivalent sharpen in albumentations compared to F.adjust_sharpness(Image.fromarray(img.copy()), val)
        # Replaced by Unsharpen mask + blure
        # A.UnsharpMask(sigma_limit=sharpen[M], alpha=(0.2, np.clip(sharpen, 0.2, 1.0)), p=p),
        A.UnsharpMask(sigma_limit=sharpen[M], p=p),
        A.Blur(blur_limit=3, p=p),
    ]
    """
    Geometric Transformations
    """
    # Shear X: 0=no shear, ~17=max degree, Range [-shear_x,shear_x]
    shear_x = np.linspace(0, np.degrees(np.arctan(0.3)), bins)
    # Shear Y: 0=no shear, ~17=max degree, Range [-shear_y,shear_y]
    shear_y = np.linspace(0, np.degrees(np.arctan(0.3)), bins)
    # Translate X: 0=no translation 0.2=max translation in %, Range [-translate_x,translate_x]
    translate_x = np.linspace(0, 0.2, bins)
    # Translate Y: 0=no translation 0.2=max translation in %, Range [-translate_y,translate_y]
    translate_y = np.linspace(0, 0.2, bins)
    # Rotate: 0=no rotation, 30=rotationabout 30 degree, Range [-rotation,rotation]
    rotation = np.linspace(0, 30, bins)

    geo_transforms = [
        # Shear X - equivalen to F.affine(Image.fromarray(img.copy()),angle=0.0,translate=[0, 0],scale=1.0,shear=[math.degrees(math.atan(val)),0.0],interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,center=[0, 0],)
        A.Affine(shear={"x": (-shear_x[M], shear_x[M]), "y": 0}, p=p, mode=mode),
        # Shear Y - equivalen to F.affine(Image.fromarray(img.copy()),angle=0.0,translate=[0, 0],scale=1.0,shear=[0.0, math.degrees(math.atan(val))],interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,center=[0, 0],)
        A.Affine(shear={"x": 0, "y": (-shear_y[M], shear_y[M])}, p=p, mode=mode),
        # Translate X - F.affine(Image.fromarray(img.copy()),angle=0.0,translate=[150, 0],scale=1.0,shear=[0.0, 0.0],interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,center=[0, 0],)
        A.Affine(
            translate_percent={"x": (-translate_x[M], translate_x[M]), "y": 0}, p=p, mode=mode
        ),
        # Translate Y - F.affine(Image.fromarray(img.copy()),angle=0.0,translate=[0,150],scale=1.0,shear=[0.0, 0.0],interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,center=[0, 0],)
        A.Affine(
            translate_percent={"x": 0, "y": (-translate_y[M], translate_y[M])}, p=p, mode=mode
        ),
        # Rotate - equivalent to F.rotate(Image.fromarray(img.copy()),val,interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,)
        A.Affine(rotate=(-rotation[M], rotation[M]), p=p, mode=mode),
    ]

    """
    Return RandAugment pipeline
    """
    transforms = A.SomeOf(col_transforms + geo_transforms, n=N)
    return transforms


def RandAugment_Histo(N: int, M: int, p: float = 0.5, bins: int = 10, mode: int = 0) -> object:

    """
    https://towardsdatascience.com/augmentation-methods-using-albumentations-and-pytorch-35cd135382f8
    # https://openreview.net/pdf?id=JrBfXaoxbA2
    # https://arxiv.org/pdf/1805.09501.pdf
    Parameters
    ----------
    N: int
        Number of Operations
    M: int
        Magnitude, intensity of the operations
    p: float, optional
        Probability for each operation to be applied
    bins: int, optional
        Number of Bins, max value for M
    mode: int, optional
        Border Mode for geometric transforms
        BORDER_CONSTANT    = 0,
        BORDER_REPLICATE   = 1,
        BORDER_REFLECT     = 2,
        BORDER_WRAP        = 3,
        BORDER_REFLECT_101 = 4,
        BORDER_TRANSPARENT = 5,
        BORDER_REFLECT101  = BORDER_REFLECT_101,
        BORDER_DEFAULT     = BORDER_REFLECT_101,
        BORDER_ISOLATED    = 16,
    """
    """
    Color Transformations
    """
    # Contrast: 1=original, 0=gray, Range [max(0, 1 - contrast), 1 + contrast]
    contrast = np.linspace(0.0, 0.9, bins)
    # Brightness: 1=original, 0=black, Range [max(0, 1 - brightness), 1 + brightness]
    brightness = np.linspace(0.0, 0.9, bins)
    # Color: 1=original, 0=black-white image, Range [max(0, 1 - color), 1 + color]
    color = np.linspace(0.0, 0.9, bins)
    # Solariize: 255=original, 0=inverted, Range [solarize,255]
    solarize = np.linspace(255, 0, bins)
    # Posterize: Range [8,4]
    posterize = (8 - (np.arange(bins) / ((bins - 1) / 4)).round()).astype(int)
    # Sharpen
    sharpen = np.linspace(0.0, 2.0, bins)
    # HSV Shift
    hsv = np.linspace(0.0, 0.9, bins)

    col_transforms = [
        # Identity
        A.NoOp(p=p),
        # Contrast - equivalent to F.adjust_contrast(Image.fromarray(img), val)
        A.ColorJitter(brightness=0, contrast=contrast[M], saturation=0, hue=0, p=p),
        # Brightness - equivalent to F.adjust_brightness(Image.fromarray(img), val)
        A.ColorJitter(brightness=brightness[M], contrast=0, saturation=0, hue=0, p=p),
        # Color (~Saturation) - equivalent to F.adjust_saturation(Image.fromarray(img), val)
        A.ColorJitter(brightness=0, contrast=0, saturation=color[M], hue=0, p=p),
        # Solarize - eqilvalent to F.solarize(Image.fromarray(img), int(val))
        # A.Solarize(threshold=(solarize[M], 255), p=p),
        # Posterize - F.posterize(Image.fromarray(img), int(val))
        # A.Posterize(num_bits=int(posterize[M]), p=p),
        # Equalize - equivalent to F.equalize(Image.fromarray(img))
        # A.Equalize(mode="pil", p=p),
        # Invert - equalize to F.invert(Image.fromarray(img))
        # A.InvertImg(p=p),
        # Sharpen - no equivalent sharpen in albumentations compared to F.adjust_sharpness(Image.fromarray(img.copy()), val)
        # Replaced by Unsharpen mask + blure
        # A.UnsharpMask(sigma_limit=sharpen[M], alpha=(0.2, np.clip(sharpen, 0.2, 1.0)), p=p),
        A.UnsharpMask(sigma_limit=sharpen[M], p=p),
        A.Blur(blur_limit=3, p=p),
        JPEGcompressor(p=p),
        A.ColorJitter(
            brightness=0,
            contrast=0,
            saturation=hsv[M],
            hue=hsv[M],
            p=p,
        ),
    ]
    """
    Geometric Transformations
    """
    # Shear X: 0=no shear, ~17=max degree, Range [-shear_x,shear_x]
    shear_x = np.linspace(0, np.degrees(np.arctan(0.3)), bins)
    # Shear Y: 0=no shear, ~17=max degree, Range [-shear_y,shear_y]
    shear_y = np.linspace(0, np.degrees(np.arctan(0.3)), bins)
    # Translate X: 0=no translation 0.2=max translation in %, Range [-translate_x,translate_x]
    translate_x = np.linspace(0, 0.2, bins)
    # Translate Y: 0=no translation 0.2=max translation in %, Range [-translate_y,translate_y]
    translate_y = np.linspace(0, 0.2, bins)
    # Rotate: 0=no rotation, 30=rotationabout 30 degree, Range [-rotation,rotation]
    rotation = np.linspace(0, 30, bins)

    geo_transforms = [
        # Shear X - equivalen to F.affine(Image.fromarray(img.copy()),angle=0.0,translate=[0, 0],scale=1.0,shear=[math.degrees(math.atan(val)),0.0],interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,center=[0, 0],)
        A.Affine(shear={"x": (-shear_x[M], shear_x[M]), "y": 0}, p=p, mode=mode),
        # Shear Y - equivalen to F.affine(Image.fromarray(img.copy()),angle=0.0,translate=[0, 0],scale=1.0,shear=[0.0, math.degrees(math.atan(val))],interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,center=[0, 0],)
        A.Affine(shear={"x": 0, "y": (-shear_y[M], shear_y[M])}, p=p, mode=mode),
        # Translate X - F.affine(Image.fromarray(img.copy()),angle=0.0,translate=[150, 0],scale=1.0,shear=[0.0, 0.0],interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,center=[0, 0],)
        A.Affine(
            translate_percent={"x": (-translate_x[M], translate_x[M]), "y": 0}, p=p, mode=mode
        ),
        # Translate Y - F.affine(Image.fromarray(img.copy()),angle=0.0,translate=[0,150],scale=1.0,shear=[0.0, 0.0],interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,center=[0, 0],)
        A.Affine(
            translate_percent={"x": 0, "y": (-translate_y[M], translate_y[M])}, p=p, mode=mode
        ),
        # Rotate - equivalent to F.rotate(Image.fromarray(img.copy()),val,interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=None,)
        A.Affine(rotate=(-rotation[M], rotation[M]), p=p, mode=mode),
    ]

    """
    Return RandAugment pipeline
    """
    transforms = A.SomeOf(col_transforms + geo_transforms, n=N)
    return transforms


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
        # LightAugmentor(p=p),
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
        # JPEGcompressor(p=p),
        # HED Augmentations
        # LightAugmentor(p=p),
        # StrongAugmentor(p=p),
        # Equalize
        # A.Equalize(p=p),
    ]
    transforms = A.SomeOf(color_Aug, n=N)

    return transforms
