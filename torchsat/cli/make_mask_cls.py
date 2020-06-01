"""
 * @author sshuair
 * @email sshuair@gmail.com
 * @create date 2020-05-31 16:06:19
 * @modify date 2020-05-31 16:06:19
 * @desc this tool is to patch the large satellite image to small image
"""

import os
from glob import glob
from pathlib import Path

import click
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def patch_image(filepath, width, height, drop_last, outpath):
    with rasterio.open(filepath, 'r') as src:
        rows = src.meta['height'] // height if drop_last else src.meta['height'] // height + 1
        columns = src.meta['width'] // width if drop_last else src.meta['width'] // width + 1
        for row in tqdm(range(rows)):
            for col in range(columns):
                outfile = os.path.join(outpath, Path(filepath).stem+'_'+str(row)+'_'+str(col)+Path(filepath).suffix)
                window = Window(col * width, row * height, width, height)
                patched_arr = src.read(window=window, boundless=True)
                kwargs = src.meta.copy()
                kwargs.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, src.transform)})
                with rasterio.open(outfile, 'w', **kwargs) as dst:
                    dst.write(patched_arr)


@click.command(help='this tool is to patch the large satellite image to small image.')
@click.option('--filepath', type=str, help='the target satellite image to split. Note: the file should have crs')
@click.option('--width', default=256, type=int, help='the width of the patched image')
@click.option('--height', default=256, type=int, help='the height of the patched image')
@click.option('--drop_last', default=True, type=bool,
              help='set to True to drop the last column and row, if the image size is not divisible by the height and width.')
@click.option('--outpath', type=str, help='the output file path')
def make_mask_cls(filepath: str, width: int, height: int, drop_last: bool, outpath: str):
    if Path(filepath).is_file():
        files = [filepath]
    else:
        files = glob(os.path.join(filepath, '**/*.tif'), recursive=True)
    for idx, item in enumerate(files):
        print('processing {}/{} file {} ...'.format(idx + 1, len(files), item))
        patch_image(item, width, height, drop_last, outpath)


if __name__ == "__main__":
    main()
