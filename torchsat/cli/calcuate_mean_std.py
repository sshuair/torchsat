import argparse
import os
import random
from glob import glob
from pathlib import Path

import click
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


@click.command(help='calcuate the datasets mean and std value')
@click.option('--root', type=click.Path(exists=True), required=True, help='root dir of image datasets')
@click.option("--percent", default=0.5, type=float, help="percent of images to calcuate")
@click.option("--channels", default=3, type=int, help="datasets image channels")
@click.option("--maxvalue", default=255, type=float, help="max value of all images default: {255}")
@click.option("--extension", type=str, default=('jpg', 'jpeg', 'png', 'tif', 'tiff'), multiple=True,
              help="file suffix to calcuate, default ('jpg', 'jpeg', 'png', 'tif', 'tiff')")
def calcuate_mean_std(root, percent, channels, maxvalue, extension):
    files = [x for x in Path(root).glob('**/*') if x.suffix.lower()[1:] in extension and '._' not in str(x)]
    random.shuffle(files)
    files = files[0: int(len(files) * percent)]

    if not files:
        print("INFO: No Image Found!")
        return

    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(channels)
    channel_sum_squared = np.zeros(channels)
    for item in tqdm(files):
        arr = img_loader(item)
        arr = arr / maxvalue
        pixel_num += arr.shape[0] * arr.shape[1]
        channel_sum += np.sum(arr, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(arr), axis=(0, 1))

    mean = channel_sum / pixel_num
    std = np.sqrt(channel_sum_squared / pixel_num - np.square(mean))

    print("scaled  mean:{} \nscaled  std: {} ".format(mean, std))
    print("orginal mean: {} \norginal std: {}".format(mean * maxvalue, std * maxvalue))

if __name__ == "__main__":
    calcuate_mean_std()