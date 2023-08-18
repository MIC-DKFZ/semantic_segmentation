import numpy as np
import albumentations as A


def RandAugment(N: int, M: int, p: float = 0.5, bins: int = 10, mode: int = 0) -> object:

    """
    https://towardsdatascience.com/augmentation-methods-using-albumentations-and-pytorch-35cd135382f8
    # https://openreview.net/pdf?id=JrBfXaoxbA2
    # https://arxiv.org/pdf/1805.09501.pdf
    https://arxiv.org/pdf/1909.13719.pdf,
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


def RandAugment_light(N: int, M: int, p: float = 0.5, bins: int = 10, mode: int = 0) -> object:

    """
    subset of color augmentations of RandAugment
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
