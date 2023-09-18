import os
import glob
import cv2
import multiprocessing
import argparse

ignore_label = 255
label_mapping = {
    -1: ignore_label,
    0: ignore_label,
    1: ignore_label,
    2: ignore_label,
    3: ignore_label,
    4: ignore_label,
    5: ignore_label,
    6: ignore_label,
    7: 0,
    8: 1,
    9: ignore_label,
    10: ignore_label,
    11: 2,
    12: 3,
    13: 4,
    14: ignore_label,
    15: ignore_label,
    16: ignore_label,
    17: 5,
    18: ignore_label,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: ignore_label,
    30: ignore_label,
    31: 16,
    32: 17,
    33: 18,
}


def process_file(file):
    outfile = file.split(".png")[0] + "_19classes.png"
    mask_34 = cv2.imread(file, -1)
    mask_19 = mask_34.copy()
    for k, v in label_mapping.items():
        mask_19[mask_34 == k] = v
    cv2.imwrite(outfile, mask_19)


if __name__ == "__main__":
    """
    Small Script to convert the cityscapes dataset into a suitable format,
    Convert labels into the 19 class format and save them with '_19classes' postfix
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_processes", type=int, default=8)
    args = parser.parse_args()

    root = args.data_path

    splits = ["train", "val"]
    pool = multiprocessing.Pool(processes=args.num_processes)
    async_results = []

    for split in splits:
        path = os.path.join(
            root, "gtFine_trainvaltest", "gtFine", split, "*", "*gtFine_labelIds.png"
        )
        files = glob.glob(path)
        for file in files:
            async_results.append(
                pool.starmap_async(
                    process_file,
                    ((file,),),
                )
            )

    print(f"Start Processing the Data, {len(async_results)} files found")
    _ = [a.get() for a in async_results]
    pool.close()
    pool.join()
    print("Done Processing the Data")
