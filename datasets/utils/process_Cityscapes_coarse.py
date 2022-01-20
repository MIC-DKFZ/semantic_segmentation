import os
import glob
import cv2
from tqdm import tqdm
import argparse

ignore_label=255
label_mapping = {-1: ignore_label, 0: ignore_label,
                      1: ignore_label, 2: ignore_label,
                      3: ignore_label, 4: ignore_label,
                      5: ignore_label, 6: ignore_label,
                      7: 0, 8: 1, 9: ignore_label,
                      10: ignore_label, 11: 2, 12: 3,
                      13: 4, 14: ignore_label, 15: ignore_label,
                      16: ignore_label, 17: 5, 18: ignore_label,
                      19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                      25: 12, 26: 13, 27: 14, 28: 15,
                      29: ignore_label, 30: ignore_label,
                      31: 16, 32: 17, 33: 18}

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',type=str)
    args = parser.parse_args()

    root=args.data_path

    splits = ["train_extra"]

    for split in splits:
        path = os.path.join(root,"gtCoarse", "gtCoarse", split, "*", "*gtCoarse_labelIds.png")
        files = glob.glob(path)
        for file in tqdm(files):
            outfile = file.split(".png")[0] + "_19classes.png"
            mask_34 = cv2.imread(file, -1)
            mask_19 = mask_34.copy()
            for k, v in label_mapping.items():
                mask_19[mask_34 == k] = v
            cv2.imwrite(outfile, mask_19)