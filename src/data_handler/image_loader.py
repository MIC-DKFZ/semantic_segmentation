import cv2

from src.data_handler.base_handler import BaseLoader
from src.visualization.utils import show_img


class RGBLoader(BaseLoader):
    def load_file(self, file):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def viz_img(self, img, mean, std, *args, **kwargs):
        return show_img(img, mean, std, *args, **kwargs)