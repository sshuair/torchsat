from pathlib import Path
import math

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from torchsat.transforms import transforms_det

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

bboxes = np.array([
    [12, 2, 417, 467],
    [7, 39, 63, 94],
    [362, 24, 422, 53],
    [376, 36, 422, 81],
    [373, 68, 422, 108],
    [376, 98, 422, 210]
], dtype=np.float)

labels = np.array([2,2,1,3], dtype=np.int64)

def read_img(fp):
    if Path(fp).suffix in ['.tif', '.tiff']:
        img = tifffile.imread(fp)
    else:
        img = np.array(Image.open(fp))
    return img


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_ToTensor(fp):
    img = read_img(fp)
    out_img, out_bboxes, out_labels = transforms_det.Compose([
        transforms_det.ToTensor()
    ])(img, bboxes, labels)
    assert type(out_bboxes) == torch.Tensor
    assert out_bboxes.shape == bboxes.shape
    assert type(out_labels) == torch.Tensor
    assert out_labels.shape == labels.shape


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_Normalize(fp):
    img = read_img(fp)
    channels = 1 if img.ndim==2 else img.shape[2]
    mean = [img.mean()] if channels==1 else np.array(img.mean(axis=(0, 1))).tolist()
    std = [img.std()] if channels==1 else np.array(img.std(axis=(0, 1))).tolist()

    out_img, out_bboxes, out_labels = transforms_det.Compose([
        transforms_det.ToTensor(),
        transforms_det.Normalize(mean, std)
    ])(img, bboxes, labels)
    assert type(out_bboxes) == torch.Tensor
    assert out_bboxes.shape == bboxes.shape
    assert type(out_labels) == torch.Tensor
    assert out_labels.shape == labels.shape


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_ToGray(fp):
    img = read_img(fp)
    out_img, out_bboxes, out_labels = transforms_det.Compose([
        transforms_det.ToGray()
    ])(img, bboxes, labels)
    assert out_bboxes.dtype == bboxes.dtype
    assert out_bboxes.shape == bboxes.shape
    assert out_labels.dtype == labels.dtype
    assert out_labels.shape == labels.shape

    out_img, out_bboxes, out_labels = transforms_det.Compose([
        transforms_det.ToGray(output_channels=5)
    ])(img, bboxes, labels)
    assert out_bboxes.dtype == bboxes.dtype
    assert out_bboxes.shape == bboxes.shape
    assert out_labels.dtype == labels.dtype
    assert out_labels.shape == labels.shape
