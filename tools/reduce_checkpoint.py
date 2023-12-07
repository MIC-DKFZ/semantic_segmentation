import argparse
import glob
from os.path import join
import torch


def reduce_ckpt(ckpt_dir):
    ckpt_files = glob.glob(join(ckpt_dir, "*.ckpt"))
    for ckpt_file in ckpt_files:
        ckpt = torch.load(ckpt_file)["state_dict"]

        torch.save(ckpt, ckpt_file.replace(".ckpt", ".pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to the checkpoint folder to be reduced",
    )
    args = parser.parse_args()

    ckpt_dir = args.input
    reduce_ckpt(ckpt_dir)
