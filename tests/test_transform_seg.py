from pathlib import Path
import math

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from torchsat.transforms import transforms_seg

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

mask_file = './tests/fixtures/masks/mask_tiff_3channel_uint8.png'


def read_img(fp):
    if Path(fp).suffix in ['.tif', '.tiff']:
        img = tifffile.imread(fp)
    else:
        img = np.array(Image.open(fp))
    return img


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_ToTensor(fp):
    img = read_img(fp)
    mask = read_img(mask_file)
    
    result_img, result_mask = transforms_seg.Compose([
        transforms_seg.ToTensor()
    ])(img, mask)
    assert type(result_img) == torch.Tensor
    assert len(result_img.shape) == 3
    assert result_img.shape[1:3] == img.shape[0:2]

    assert type(result_mask) == torch.Tensor
    assert torch.all(torch.unique(result_mask) == torch.tensor([0,1,2,3])) == True


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_Normalize(fp):
    img = read_img(fp)
    mask = read_img(mask_file)
    channels = 1 if img.ndim==2 else img.shape[2]
    mean = [img.mean()] if channels==1 else np.array(img.mean(axis=(0, 1))).tolist()
    std = [img.std()] if channels==1 else np.array(img.std(axis=(0, 1))).tolist()

    result_img, result_mask = transforms_seg.Compose([
        transforms_seg.ToTensor(),
        transforms_seg.Normalize(mean, std)
    ])(img, mask)
    assert type(result_img) == torch.Tensor
    assert len(result_img.shape) == 3
    assert result_img.shape[1:3] == img.shape[0:2]

    assert type(result_mask) == torch.Tensor
    assert torch.all(torch.unique(result_mask) == torch.tensor([0,1,2,3])) == True


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_ToGray(fp):
    img = read_img(fp)
    mask = read_img(mask_file)
    result_img, result_mask = transforms_seg.Compose([
        transforms_seg.ToGray()
    ])(img, mask)
    assert result_img.dtype == img.dtype
    assert result_img.ndim == 2

    result_img, result_mask = transforms_seg.Compose([
        transforms_seg.ToGray(output_channels=5)
    ])(img, mask)
    assert result_img.shape == (img.shape[0], img.shape[1], 5)
    assert result_img.dtype == img.dtype

    assert result_mask.dtype == mask.dtype
    assert np.all(np.unique(result_mask) == np.array([0,1,2,3])) == True


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_GaussianBlur(fp):
    img = read_img(fp)
    mask = read_img(mask_file)
    result_img, result_mask = transforms_seg.Compose([
        transforms_seg.GaussianBlur(kernel_size=5)
    ])(img, mask)
    assert result_img.shape == img.shape
    assert result_img.dtype == img.dtype

    assert result_mask.dtype == mask.dtype
    assert np.all(np.unique(result_mask) == np.array([0,1,2,3])) == True


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomNoise(fp):
    img = read_img(fp)
    mask = read_img(mask_file)
    for item in ['gaussian', 'salt', 'pepper', 's&p']:
        result_img, result_mask = transforms_seg.Compose([
            transforms_seg.RandomNoise(mode=item)
        ])(img, mask)
        assert result_img.shape == img.shape
        assert result_img.dtype == img.dtype

        assert result_mask.dtype == mask.dtype
        assert np.all(np.unique(result_mask) == np.array([0,1,2,3])) == True


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomBrightness(fp):
    img = read_img(fp)
    mask = read_img(mask_file)
    result_img, result_mask = transforms_seg.Compose([
        transforms_seg.RandomBrightness()
    ])(img, mask)
    assert result_img.shape == img.shape
    assert result_img.dtype == img.dtype
    assert result_mask.dtype == mask.dtype
    assert np.all(np.unique(result_mask) == np.array([0,1,2,3])) == True

    result_img, result_mask = transforms_seg.Compose([
        transforms_seg.RandomBrightness(max_value=10)
    ])(img, mask)
    assert result_img.shape == img.shape
    assert result_img.dtype == img.dtype
    if result_img.ndim == 2:
        assert abs(float(result_img[0,0]) - float(img[0,0])) <=10
    else:
        assert abs(float(result_img[0,0,0]) - float(img[0,0,0])) <=10
    assert result_mask.dtype == mask.dtype
    assert np.all(np.unique(result_mask) == np.array([0,1,2,3])) == True


@pytest.mark.parametrize('fp', tiff_files+jpeg_files)
def test_RandomContrast(fp):
    img = read_img(fp)
    mask = read_img(mask_file)
    result_img, result_mask = transforms_seg.Compose([
        transforms_seg.RandomContrast()
    ])(img, mask)
    assert result_img.shape == img.shape
    assert result_img.dtype == img.dtype
    assert result_mask.dtype == mask.dtype
    assert np.all(np.unique(result_mask) == np.array([0,1,2,3])) == True

    result_img, result_mask = transforms_seg.Compose([
        transforms_seg.RandomContrast(max_factor=1.2)
    ])(img, mask)
    assert result_img.shape == img.shape
    assert result_img.dtype == img.dtype
    if result_img.ndim == 2:
        assert abs(float(result_img[0,0]) / float(img[0,0])) <=1.2
    else:
        assert abs(float(result_img[0,0,0]) / float(img[0,0,0])) <=1.2
    assert result_mask.dtype == mask.dtype
    assert np.all(np.unique(result_mask) == np.array([0,1,2,3])) == True


# @pytest.mark.parametrize('fp', tiff_files+jpeg_files)
# def test_Resize(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.Compose([
#         transforms_seg.Resize(300),
#         transforms_seg.ToTensor(),
#     ])(img)
#     assert result.shape[1:3] == torch.Size([300, 300])
#     assert type(result) == torch.Tensor

#     result_img, result_mask = transforms_seg.Compose([
#         transforms_seg.Resize(833),
#     ])(img)
#     assert result.shape[0:2] == (833, 833)
#     assert result.dtype == img.dtype

#     result_img, result_mask = transforms_seg.Compose([
#         transforms_seg.Resize((500,300)),
#     ])(img)
#     assert result.shape[0:2] == (500, 300)
#     assert result.dtype == img.dtype


# @pytest.mark.parametrize('fp', tiff_files+jpeg_files)
# def test_CenterCrop(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.Compose([
#         transforms_seg.CenterCrop(300),
#     ])(img)
#     assert result.shape[0:2] == (300,300)
#     assert result.dtype == img.dtype

#     result_img, result_mask = transforms_seg.Compose([
#         transforms_seg.CenterCrop((500,300)),
#     ])(img)
#     assert result.shape[0:2] == (500,300)
#     assert result.dtype == img.dtype

#     with pytest.raises(ValueError) as excinfo:
#         transforms_seg.CenterCrop(1000)(img)
#         assert 'the output_size should' in str(excinfo.value)


# @pytest.mark.parametrize('fp',  tiff_files+jpeg_files)
# def test_Pad(fp):
#     img = read_img(fp)
#     # constant value
#     result_img, result_mask = transforms_seg.Pad(10, fill=1)(img)
#     if result.ndim == 2:
#         assert result[0,0] == 1
#     else:
#         assert result[0,0,0] == 1
    
#     # reflect value
#     result_img, result_mask = transforms_seg.Pad(20, padding_mode='reflect')(img)
#     assert result.shape[0:2] == (img.shape[0]+40, img.shape[1]+40)
#     if result.ndim == 2:
#         assert result[0,0] == img[20,20]
#     else:
#         assert result[0,0,0] == img[20,20,0]
#     assert result.dtype == img.dtype
    
#     # all padding mode methods
#     for item in ['reflect','edge','linear_ramp','maximum','mean' , 'median', 'minimum','symmetric','wrap']:
#     # for item in ['edge']:
#         result_img, result_mask = transforms_seg.Pad(10, padding_mode=item)(img)
#         assert result.dtype == img.dtype
#         assert result.shape[0:2] == (img.shape[0]+20, img.shape[1]+20)

#         result_img, result_mask = transforms_seg.Pad((10,20), padding_mode=item)(img)
#         assert result.shape[0:2] == (img.shape[0]+40, img.shape[1]+20)
#         assert result.dtype == img.dtype

#         result_img, result_mask = transforms_seg.Pad((10,20,30,40), padding_mode=item)(img)
#         assert result.shape[0:2] == (img.shape[0]+60, img.shape[1]+40)
#         assert result.dtype == img.dtype

#     result_img, result_mask = transforms_seg.Compose([
#         transforms_seg.Pad(10, fill=1),
#         transforms_seg.ToTensor()
#     ])(img)
#     assert type(result) == torch.Tensor


# @pytest.mark.parametrize('fp',  tiff_files+jpeg_files)
# def test_RandomCrop(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.RandomCrop(111)(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == (111,111)

#     result_img, result_mask = transforms_seg.RandomCrop((100, 200))(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == (100,200)


# @pytest.mark.parametrize('fp', tiff_files+jpeg_files)
# def test_RandomHorizontalFlip(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.RandomHorizontalFlip(p=1)(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == img.shape[0:2]
#     if result.ndim == 2:
#         height, width = img.shape
#         assert result[height-1,0] == img[0,0]
#     else:
#         height, width, depth = img.shape
#         assert (result[height-1,0,:] == img[0,0,:]).any() == True

#     # tensor
#     result_img, result_mask = transforms_seg.Compose([
#         transforms_seg.RandomHorizontalFlip(p=1),
#         transforms_seg.ToTensor()
#     ])(img)
#     assert type(result) == torch.Tensor
#     assert result.shape[1:3] == img.shape[0:2]


# @pytest.mark.parametrize('fp', tiff_files+jpeg_files)
# def test_RandomVerticalFlip(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.RandomVerticalFlip(p=1)(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == img.shape[0:2]
#     if result.ndim == 2:
#         height, width = img.shape
#         assert result[0,width-1] == img[0,0]
#     else:
#         height, width, depth = img.shape
#         assert (result[0,width-1,:] == img[0,0,:]).any() == True

#     # tensor
#     result_img, result_mask = transforms_seg.Compose([
#         transforms_seg.RandomVerticalFlip(p=1),
#         transforms_seg.ToTensor()
#     ])(img)
#     assert type(result) == torch.Tensor
#     assert result.shape[1:3] == img.shape[0:2]


# @pytest.mark.parametrize('fp', tiff_files+jpeg_files)
# def test_Flip(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.RandomFlip(p=0)(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == img.shape[0:2]
#     if result.ndim == 2:
#         height, width = img.shape
#         assert result[0,0] == img[0,0]
#     else:
#         height, width, depth = img.shape
#         assert (result[0,0,:] == img[0,0,:]).any() == True

#     # tensor
#     result_img, result_mask = transforms_seg.Compose([
#         transforms_seg.RandomFlip(p=0.1),
#         transforms_seg.ToTensor()
#     ])(img)
#     assert type(result) == torch.Tensor
#     assert result.shape[1:3] == img.shape[0:2]


# @pytest.mark.parametrize('fp', tiff_files+jpeg_files)
# def test_RandomResizedCrop(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.RandomResizedCrop((500,300), 300)(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == (300,300)

#     result_img, result_mask = transforms_seg.RandomResizedCrop(500, (500,300))(img)
#     assert result.shape[0:2] == (500,300)


# @pytest.mark.parametrize('fp', tiff_files+jpeg_files)
# def test_ElasticTransform(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.ElasticTransform()(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == img.shape[0:2]


# @pytest.mark.parametrize('fp', tiff_files+jpeg_files)
# def test_RandomRotation(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.RandomRotation(45)(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == img.shape[0:2]

#     result_img, result_mask = transforms_seg.RandomRotation((-10, 30))(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == img.shape[0:2]

#     result_img, result_mask = transforms_seg.RandomRotation((-10, 30), center=(200,250))(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == img.shape[0:2]


# @pytest.mark.parametrize('fp', tiff_files+jpeg_files)
# def test_RandomShift(fp):
#     img = read_img(fp)
#     result_img, result_mask = transforms_seg.RandomShift(max_percent=0.1)(img)
#     assert result.dtype == img.dtype
#     assert result.shape[0:2] == img.shape[0:2]