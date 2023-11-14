import cv2
from src.visualization.utils import show_img


class RGBLoader:
    def load_img(self, file):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def show_img(self, img, mean, std, *args, **kwargs):
        return show_img(img, mean, std, *args, **kwargs)
