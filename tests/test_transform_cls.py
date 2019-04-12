import torchsat.transforms.transforms_cls as transforms_cls
import numpy as np
from PIL import Image
import torch

def test_resize():
    np_image_3channel = np.random.randint(0,255, size=(435, 300,3), dtype=np.uint8)
    np_image_1channel = np.random.randint(0,65535, size=(435, 300), dtype=np.uint16)

    result = transforms_cls.Compose([
        transforms_cls.Resize(300),
        transforms_cls.ToTensor(),
    ])(np_image_3channel)
    assert result.shape == torch.Size([3, 300, 300])
    assert type(result) == torch.Tensor


    result = transforms_cls.Compose([
        transforms_cls.Resize(300),
    ])(np_image_1channel)
    assert result.shape == (300, 300)
    assert type(result) == np.ndarray


