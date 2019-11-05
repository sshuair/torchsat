import argparse
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm


def img_loader(fp):
    if Path(fp).suffix.lower() in [".jpg", ".jpeg", ".png"]:
        arr = np.array(Image.open(fp))
    else:
        arr = tifffile.imread(fp)

    return arr


def main(args):
    files = glob(os.path.join(args.root, "**/*"), recursive=True)
    files = [x for x in files if Path(x).suffix.replace(".", "") in args.fmt]
    random.shuffle(files)
    files = files[0 : int(len(files) * args.percent)]

    if not files:
        print("INFO: No Image Found!")
        return

    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(args.channels)
    channel_sum_squared = np.zeros(args.channels)
    for item in tqdm(files):
        arr = img_loader(item)
        arr = arr / args.max
        pixel_num += arr.shape[0] * arr.shape[1]
        channel_sum += np.sum(arr, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(arr), axis=(0, 1))

    mean = channel_sum / pixel_num
    std = np.sqrt(channel_sum_squared / pixel_num - np.square(mean))

    print("scaled  mean:{} \nscaled  std: {} ".format(mean, std))
    print("orginal mean: {} \norginal std: {}".format(mean * args.max, std * args.max))


def parse_args():
    parser = argparse.ArgumentParser(
        description="calcuate the datasets mean and std value"
    )
    parser.add_argument(
        "--root", required=True, type=str, help="root dir of image datasets"
    )
    parser.add_argument(
        "--fmt",
        default=["jpg"],
        nargs="+",
        help="file suffix to calcuate, default{jpg}, support suffix: jpg, jpeg, png, tif, tiff",
    )
    parser.add_argument("--percent", default=0.5, help="percent of images to calcuate")
    parser.add_argument("--channels", default=3, help="datasets image channels")
    parser.add_argument(
        "--max", default=255, type=float, help="max value of all images default: {255}"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
