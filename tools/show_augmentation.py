import glob
from os.path import join
import cv2
import albumentations as A
import numpy as np

if __name__ == "__main__":
    img_root = "/media/l727r/data/UFZ_2023_Fishinspector/Dataset222_Fishinspector/imagesTr"
    idx = 0

    img_files = glob.glob(join(img_root, "*"))
    img_files.sort()
    img_file = img_files[idx]

    img = cv2.imread(img_file)

    augmentation = A.ColorJitter
    remaining_parameter = {"contrast": 0}
    target_parameter = "brightness"
    parameter_range = [0, 1]
    ranges = np.linspace(parameter_range[0], parameter_range[1], 10)

    augmentation = A.Emboss
    remaining_parameter = {}
    target_parameter = "strength"
    parameter_range = [0, 2]
    ranges = np.linspace(parameter_range[0], parameter_range[1], 10)

    augmentation = A.Equalize  # +  A.RandomSunFlare, RandomToneCurve, A.CLAHE
    remaining_parameter = {}
    target_parameter = ""
    parameter_range = [0, 0]
    ranges = np.linspace(parameter_range[0], parameter_range[1], 10)

    augmentation = A.GridDropout
    remaining_parameter = {}
    target_parameter = ""
    parameter_range = [0, 200]
    ranges = np.linspace(parameter_range[0], parameter_range[1], 10)

    img_fig = img.copy()
    for r in ranges:
        if target_parameter != "":
            transform = A.Compose(
                augmentation(**{target_parameter: (r, r)}, **remaining_parameter, p=1)
            )
        else:
            transform = A.Compose(augmentation(**remaining_parameter, p=1))
        img_trans = transform(image=img)["image"]
        img_trans = cv2.putText(
            img_trans,
            f"{r:.2f}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        img_fig = cv2.vconcat([img_fig, img_trans])

    cv2.namedWindow("Fullscreen Window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Fullscreen Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Fullscreen Window", img_fig)
    cv2.waitKey(0)
