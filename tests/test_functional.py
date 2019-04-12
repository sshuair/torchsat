from functools import wraps
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

import torchsat.transforms.functional as F

tiff_files = [
    './tests/fixtures/different-types/tiff_1channel_float.tif',
    './tests/fixtures/different-types/tiff_1channel_uint16.tif',
    './tests/fixtures/different-types/tiff_1channel_uint8.tif',
    './tests/fixtures/different-types/tiff_3channel_float.tif',
    './tests/fixtures/different-types/tiff_3channel_uint16.tif',
    './tests/fixtures/different-types/tiff_3channel_uint8.tif',
    './tests/fixtures/different-types/tiff_8channel_float.tif',
    './tests/fixtures/different-types/tiff_8channel_uint16.tif',
    './tests/fixtures/different-types/tiff_8channel_uint8.tif',
]

jpeg_files = [
    './tests/fixtures/different-types/jpeg_1channel_uint8.jpeg',
    './tests/fixtures/different-types/jpeg_3channel_uint8.jpeg',
    './tests/fixtures/different-types/jpeg_1channel_uint8.png',
    './tests/fixtures/different-types/jpeg_3channel_uint8.png',
]

all_files = tiff_files+jpeg_files

@pytest.mark.parametrize('fp', all_files)
def test_resize(fp):
    if Path(fp).suffix in ['.tif', '.tiff']:
        src_img = tifffile.imread(fp)
    else:
        src_img = np.array(Image.open(fp))

    size = 256
    dst_img = F.resize(src_img, size)
    assert dst_img.shape[0:2] == (256,256)

    size = (128, 256)
    dst_img = F.resize(src_img, size)
    assert dst_img.shape[0:2] == (128,256)


@pytest.mark.parametrize('fp', all_files)
def test_pad(fp):
    if Path(fp).suffix in ['.tif', '.tiff']:
        img = tifffile.imread(fp)
    else:
        img = np.array(Image.open(fp))

    padding = 50

    dst = F.pad(img, padding)
    assert dst.shape[0:2] == (750, 600)

    if len(img.shape) == 2:
        assert dst[0,0] == img[50,50]
    if len(img.shape) == 3:
        assert dst[0,0,0] == img[50,50,0]
    

# @pytest.mask.parametrize('fp', files)
# def test_crop(fp):
#     left, top, width, height = 2, 2, 256, 256
#     # src = tifffile.
