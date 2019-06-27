from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from torchsat.transforms import transforms_cls

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

single_channel_files = [
    './tests/fixtures/different-types/tiff_1channel_float.tif',
    './tests/fixtures/different-types/tiff_1channel_uint16.tif',
    './tests/fixtures/different-types/tiff_1channel_uint8.tif',
    './tests/fixtures/different-types/jpeg_1channel_uint8.jpeg',
    './tests/fixtures/different-types/jpeg_1channel_uint8.png',
]


def read_img(fp):
    if Path(fp).suffix in ['.tif', '.tiff']:
        img = tifffile.imread(fp)
    else:
        img = np.array(Image.open(fp))
    return img


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_ToTensor(fp):
    img = read_img(fp)
    result = transforms_cls.Compose([
        transforms_cls.ToTensor()
    ])(img)
    assert type(result) == torch.Tensor
    assert len(result.shape) == 3
    assert result.shape[1:3] == img.shape[0:2]


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_Normalize(fp):
    img = read_img(fp)
    channels = 1 if img.ndim==2 else img.shape[2]
    mean = [img.mean()] if channels==1 else np.array(img.mean(axis=(0, 1))).tolist()
    std = [img.std()] if channels==1 else np.array(img.std(axis=(0, 1))).tolist()

    result = transforms_cls.Compose([
        transforms_cls.ToTensor(),
        transforms_cls.Normalize(mean, std)
    ])(img)
    assert type(result) == torch.Tensor
    assert len(result.shape) == 3
    assert result.shape[1:3] == img.shape[0:2]


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_ToGray(fp):
    img = read_img(fp)
    result = transforms_cls.Compose([
        transforms_cls.ToGray()
    ])(img)
    assert result.dtype == img.dtype
    assert result.ndim == 2

    result = transforms_cls.Compose([
        transforms_cls.ToGray(output_channels=5)
    ])(img)
    assert result.shape == (img.shape[0], img.shape[1], 5)
    assert result.dtype == img.dtype


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_GaussianBlur(fp):
    img = read_img(fp)
    result = transforms_cls.Compose([
        transforms_cls.GaussianBlur(kernel_size=5)
    ])(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomNoise(fp):
    img = read_img(fp)
    for item in ['gaussian', 'salt', 'pepper', 's&p']:
        result = transforms_cls.Compose([
            transforms_cls.RandomNoise(mode=item)
        ])(img)
        assert result.shape == img.shape
        assert result.dtype == img.dtype



@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_resize(fp):
    img = read_img(fp)
    result = transforms_cls.Compose([
        transforms_cls.Resize(300),
        transforms_cls.ToTensor(),
    ])(img)
    assert result.shape[1:3] == torch.Size([300, 300])

    result = transforms_cls.Compose([
        transforms_cls.Resize(833),
        transforms_cls.ToTensor(),
    ])(img)
    assert result.shape[1:3] == torch.Size([833, 833])