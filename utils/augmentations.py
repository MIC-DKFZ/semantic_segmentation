import random
import numpy as np

from io import BytesIO
from PIL import Image
from albumentations.core.transforms_interface import ImageOnlyTransform
from utils.staintools.stain_augmentor import StainAugmentor

light_augmentor = StainAugmentor(method="vahadane", sigma1=0.2, sigma2=0.2)
strong_augmentor = StainAugmentor(method="vahadane", sigma1=-0.8, sigma2=-0.8)


class LightAugmentor(ImageOnlyTransform):
    def apply(self, img, **params):
        return VahadaneLightAugmentor(img)


class StrongAugmentor(ImageOnlyTransform):
    def apply(self, img, **params):
        return VahadaneStrongAugmentor(img)


class VahadaneNormalizer(ImageOnlyTransform):
    def apply(self, img, **params):
        return Vahadane_Normalizer(img)


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
