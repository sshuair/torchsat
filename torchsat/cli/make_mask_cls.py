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

color_map = {
    0: (0, 0, 0, 255),
    1: (31, 119, 180, 255),
    2: (174, 199, 232, 255),
    3: (255, 127, 14, 255),
    4: (255, 187, 120, 255),
    5: (44, 160, 44, 255),
    6: (152, 223, 138, 255),
    7: (214, 39, 40, 255),
    8: (255, 152, 150, 255),
    9: (148, 103, 189, 255),
    10: (197, 176, 213, 255),
    11: (140, 86, 75, 255),
    12:(196, 156, 148, 255),
    13: (227, 119, 194, 255),
    14: (247, 182, 210, 255),
    15: (127, 127, 127, 255),
    16: (199, 199, 199, 255),
    17: (188, 189, 34, 255),
    18: (219, 219, 141, 255),
    19: (23, 190, 207, 255),
    20: (158, 218, 229)
}


def patch_image(filepath, width, height, drop_last, outpath, colormap):
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
                    if colormap:
                        dst.write_colormap(1, color_map)


@click.command(help='this tool is to patch the large satellite image to small image.')
@click.option('--filepath', type=str, help='the target satellite image to split. Note: the file should have crs')
@click.option('--width', default=512, type=int, help='the width of the patched image, default 512')
@click.option('--height', default=512, type=int, help='the height of the patched image, default 512')
@click.option('--drop_last', default=True, type=bool,
              help='set to True to drop the last column and row, if the image size is not divisible by the height and width. default True.')
@click.option('--outpath', type=str, help='the output file path')
@click.option('--colormap', type=bool, default=False, help='weather write the colormap to the output patched image. only work for single channel image. default False.')
@click.option('--extension', type=str, multiple=True, default=('jpg', 'jpeg', 'png', 'tif', 'tiff'), help='the train image extension, only work for dirctory file path.')
def make_mask_cls(filepath: str, width: int, height: int, drop_last: bool, outpath: str, colormap: bool, extension:tuple):
    print(filepath)
    if Path(filepath).is_file():
        files = [filepath]
    else:
        files = [x for x in Path(filepath).glob('**/*') if x.suffix.lower()[1:] in extension and '._' not in str(x)]
    for idx, item in enumerate(files):
        print('processing {}/{} file {} ...'.format(idx + 1, len(files), item))
        patch_image(item, width, height, drop_last, outpath, colormap)

