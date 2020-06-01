"""
 * @author sshuair
 * @email sshuair@gmail.com
 * @create date 2020-05-31 16:06:19
 * @modify date 2020-05-31 21:15:30
 * @desc this tool is to patch the large satellite image to small image and label for segmentation.
"""


import os
from glob import glob
from pathlib import Path

import click
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from tqdm import tqdm
import geopandas
from shapely.geometry import Polygon


@click.command(help='this tool is to patch the large satellite image to small image and label for segmentation.')
@click.option('--image_file', type=str, help='the target satellite image to split. Note: the file should have crs')
@click.option('--label_file', type=str, help='''the corresponding label file of the satellite image. 
                vector or raster file. Note the crs should be same as satellite image.''')
@click.option('--field', type=str, help='field to burn')
@click.option('--width', default=256, type=int, help='the width of the patched image')
@click.option('--height', default=256, type=int, help='the height of the patched image')
@click.option('--drop_last', default=True, type=bool,
              help='set to True to drop the last column and row, if the image size is not divisible by the height and width.')
@click.option('--outpath', type=str, help='the output file path')
def make_mask_seg(image_file: str, label_file: str, field, width: int, height: int, drop_last: bool, outpath: str):
    if not Path(image_file).is_file():
        raise ValueError('file {} not exits.'.format(image_file))
    # TODO: Check the crs

    # read the file and distinguish the label_file is raster or vector
    try:
        label_src = rasterio.open(label_file)
        label_flag = 'raster'
    except rasterio.RasterioIOError:
        label_df = geopandas.read_file(label_file)
        # TODO: create spatial index to speed up the clip
        label_flag = 'vector'

    img_src = rasterio.open(image_file)
    rows = img_src.meta['height'] // height if drop_last else img_src.meta['height'] // height + 1
    columns = img_src.meta['width'] // width if drop_last else img_src.meta['width'] // width + 1
    for row in tqdm(range(rows)):
        for col in range(columns):
            # image
            outfile_image = os.path.join(outpath, Path(image_file).stem+'_'+str(row)+'_'+str(col)+Path(image_file).suffix)
            window = Window(col * width, row * height, width, height)
            patched_arr = img_src.read(window=window, boundless=True)
            kwargs = img_src.meta.copy()
            patched_transform = rasterio.windows.transform(window, img_src.transform)
            kwargs.update({
                'height': window.height,
                'width': window.width,
                'transform': patched_transform})
            with rasterio.open(outfile_image, 'w', **kwargs) as dst:
                dst.write(patched_arr)

            # label
            outfile_label = Path(outfile_image).with_suffix('.png')
            if label_flag == 'raster':
                label_arr = label_src.read(window=window, boundless=True)
            else:
                bounds = rasterio.windows.bounds(window, img_src.transform)
                clipped_poly = geopandas.clip(label_df, Polygon.from_bounds(*bounds))
                shapes = [(geom, value) for geom, value in zip(clipped_poly.geometry, clipped_poly[field])]
                label_arr = rasterize(shapes, out_shape=(width, height), default_value=0, transform=patched_transform)   
            
            kwargs = img_src.meta.copy()
            kwargs.update({
                'driver': 'png',
                'count': 1,
                'height': window.height,
                'width': window.width,
                'transform': patched_transform,
                'dtype': 'uint8'
            })
            with rasterio.open(outfile_label, 'w', **kwargs) as dst:
                dst.write(label_arr, 1)

    img_src.close()


if __name__ == "__main__":
    main()
