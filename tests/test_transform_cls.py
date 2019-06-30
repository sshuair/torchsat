from pathlib import Path
import math

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
def test_RandomBrightness(fp):
    img = read_img(fp)
    result = transforms_cls.Compose([
        transforms_cls.RandomBrightness()
    ])(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype

    result = transforms_cls.Compose([
        transforms_cls.RandomBrightness(max_value=10)
    ])(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    if result.ndim == 2:
        assert abs(float(result[0,0]) - float(img[0,0])) <=10
    else:
        assert abs(float(result[0,0,0]) - float(img[0,0,0])) <=10


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomContrast(fp):
    img = read_img(fp)
    result = transforms_cls.Compose([
        transforms_cls.RandomContrast()
    ])(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype

    result = transforms_cls.Compose([
        transforms_cls.RandomContrast(max_factor=1.2)
    ])(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    if result.ndim == 2:
        assert abs(float(result[0,0]) / float(img[0,0])) <=1.2
    else:
        assert abs(float(result[0,0,0]) / float(img[0,0,0])) <=1.2


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_Resize(fp):
    img = read_img(fp)
    result = transforms_cls.Compose([
        transforms_cls.Resize(300),
        transforms_cls.ToTensor(),
    ])(img)
    assert result.shape[1:3] == torch.Size([300, 300])
    assert type(result) == torch.Tensor

    result = transforms_cls.Compose([
        transforms_cls.Resize(833),
    ])(img)
    assert result.shape[0:2] == (833, 833)
    assert result.dtype == img.dtype

    result = transforms_cls.Compose([
        transforms_cls.Resize((500,300)),
    ])(img)
    assert result.shape[0:2] == (500, 300)
    assert result.dtype == img.dtype


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_CenterCrop(fp):
    img = read_img(fp)
    result = transforms_cls.Compose([
        transforms_cls.CenterCrop(300),
    ])(img)
    assert result.shape[0:2] == (300,300)
    assert result.dtype == img.dtype

    result = transforms_cls.Compose([
        transforms_cls.CenterCrop((500,300)),
    ])(img)
    assert result.shape[0:2] == (500,300)
    assert result.dtype == img.dtype

    with pytest.raises(ValueError) as excinfo:
        transforms_cls.CenterCrop(1000)(img)
        assert 'the output_size should' in str(excinfo.value)


@pytest.mark.parametrize('fp',  tiff_files+jpeg_files)
def test_Pad(fp):
    img = read_img(fp)
    # constant value
    result = transforms_cls.Pad(10, fill=1)(img)
    if result.ndim == 2:
        assert result[0,0] == 1
    else:
        assert result[0,0,0] == 1
    
    # reflect value
    result = transforms_cls.Pad(20, padding_mode='reflect')(img)
    assert result.shape[0:2] == (img.shape[0]+40, img.shape[1]+40)
    if result.ndim == 2:
        assert result[0,0] == img[20,20]
    else:
        assert result[0,0,0] == img[20,20,0]
    assert result.dtype == img.dtype
    
    # all padding mode methods
    for item in ['reflect','edge','linear_ramp','maximum','mean' , 'median', 'minimum','symmetric','wrap']:
    # for item in ['edge']:
        result = transforms_cls.Pad(10, padding_mode=item)(img)
        assert result.dtype == img.dtype
        assert result.shape[0:2] == (img.shape[0]+20, img.shape[1]+20)

        result = transforms_cls.Pad((10,20), padding_mode=item)(img)
        assert result.shape[0:2] == (img.shape[0]+40, img.shape[1]+20)
        assert result.dtype == img.dtype

        result = transforms_cls.Pad((10,20,30,40), padding_mode=item)(img)
        assert result.shape[0:2] == (img.shape[0]+60, img.shape[1]+40)
        assert result.dtype == img.dtype

    result = transforms_cls.Compose([
        transforms_cls.Pad(10, fill=1),
        transforms_cls.ToTensor()
    ])(img)
    assert type(result) == torch.Tensor


@pytest.mark.parametrize('fp',  tiff_files+jpeg_files)
def test_RandomCrop(fp):
    img = read_img(fp)
    result = transforms_cls.RandomCrop(111)(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == (111,111)

    result = transforms_cls.RandomCrop((100, 200))(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == (100,200)


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomHorizontalFlip(fp):
    img = read_img(fp)
    result = transforms_cls.RandomHorizontalFlip(p=1)(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == img.shape[0:2]
    if result.ndim == 2:
        height, width = img.shape
        assert result[height-1,0] == img[0,0]
    else:
        height, width, depth = img.shape
        assert (result[height-1,0,:] == img[0,0,:]).any() == True

    # tensor
    result = transforms_cls.Compose([
        transforms_cls.RandomHorizontalFlip(p=1),
        transforms_cls.ToTensor()
    ])(img)
    assert type(result) == torch.Tensor
    assert result.shape[1:3] == img.shape[0:2]


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomVerticalFlip(fp):
    img = read_img(fp)
    result = transforms_cls.RandomVerticalFlip(p=1)(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == img.shape[0:2]
    if result.ndim == 2:
        height, width = img.shape
        assert result[0,width-1] == img[0,0]
    else:
        height, width, depth = img.shape
        assert (result[0,width-1,:] == img[0,0,:]).any() == True

    # tensor
    result = transforms_cls.Compose([
        transforms_cls.RandomVerticalFlip(p=1),
        transforms_cls.ToTensor()
    ])(img)
    assert type(result) == torch.Tensor
    assert result.shape[1:3] == img.shape[0:2]


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_Flip(fp):
    img = read_img(fp)
    result = transforms_cls.RandomFlip(p=0)(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == img.shape[0:2]
    if result.ndim == 2:
        height, width = img.shape
        assert result[0,0] == img[0,0]
    else:
        height, width, depth = img.shape
        assert (result[0,0,:] == img[0,0,:]).any() == True

    # tensor
    result = transforms_cls.Compose([
        transforms_cls.RandomFlip(p=0.1),
        transforms_cls.ToTensor()
    ])(img)
    assert type(result) == torch.Tensor
    assert result.shape[1:3] == img.shape[0:2]


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomResizedCrop(fp):
    img = read_img(fp)
    result = transforms_cls.RandomResizedCrop((500,300), 300)(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == (300,300)

    result = transforms_cls.RandomResizedCrop(500, (500,300))(img)
    assert result.shape[0:2] == (500,300)


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_ElasticTransform(fp):
    img = read_img(fp)
    result = transforms_cls.ElasticTransform()(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == img.shape[0:2]


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomRotation(fp):
    img = read_img(fp)
    result = transforms_cls.RandomRotation(45)(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == img.shape[0:2]

    result = transforms_cls.RandomRotation((-10, 30))(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == img.shape[0:2]

    result = transforms_cls.RandomRotation((-10, 30), center=(200,250))(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == img.shape[0:2]


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomShift(fp):
    img = read_img(fp)
    result = transforms_cls.RandomShift(max_percent=0.1)(img)
    assert result.dtype == img.dtype
    assert result.shape[0:2] == img.shape[0:2]