import random
import numpy as np

from io import BytesIO
from PIL import Image
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


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


