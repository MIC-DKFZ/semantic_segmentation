import csv
import pandas as pd
import os
import cv2
import numpy as np

if __name__ == "__main__":
    path = "/home/l727r/Documents/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/HZDR_2022_Solar_Hydrogen"
    df = pd.read_csv(open(os.path.join(path, "annotations.csv")))
    # print(df)
    "Data/images"
    files = df.file.unique()
    print("{} unique Files are found in csv".format(len(files)))
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Window", cv2.WINDOW_AUTOSIZE)
    # cv2.resizeWindow("Window", 1200, 12d00)

    # for file in files:
    i = 0
    while True:
        file = files[i]
        rows = df[df["file"] == file]
        print("File: {}  with {} Instances".format(file, len(rows)))
        img = cv2.imread(os.path.join(path, "Data", "images", file + ".tif"))

        for x, y, r in zip(rows.x, rows.y, rows.radius):
            cv2.circle(img, (int(x), int(y)), int(r), [0, 0, 255], 1)

        img *= 10
        cv2.imshow("Window", img)
        k = cv2.waitKey()
        if k == 100:
            i = min(i + 1, len(files) - 1)
        elif k == 97:
            i = max(0, i - 1)
        elif k == 113:
            break
    cv2.destroyAllWindows()

# print(df.file.unique().shape)
# for index, row in df.iterrows():
#    x, y = row.x, row.y
#    r = row.radius
#    file = row.file
#    print(file, x, y, r)
#    # break
