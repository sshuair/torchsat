import collections
import numbers
import random

import torch
import cv2
import numpy as np
from PIL import Image

from . import functional as F

__all__ = ["Compose", "Lambda", "ToTensor", "Normalize", "ToGray", "GaussianBlur",
           "RandomNoise", "RandomBrightness", "RandomContrast", "RandomShift", 
           "RandomRotation", "Resize", "Pad", "CenterCrop", "RandomCrop",
            "RandomHorizontalFlip", "RandomVerticalFlip", "RandomFlip",
           "RandomResizedCrop", "ElasticTransform",]

# 模型输入参数
# img, batch_size, channel, height, width
# boxes(float32), batch_size, bbox_nums(4个数值), 4
# labels(int64), batch_size, bbox_nums(每个box对应的类别)

# 同常有三种情况，
# 一种是三个合在一起  
# 二种是img, (boxes, labels)


class Compose(object):
    """Composes serveral classification transform together.
    
    Args:
        transforms (list of ``transform`` objects): list of classification transforms to compose.
    
    Example:
        >>> transforms_cls.Compose([
        >>>     transforms_cls.Resize(300),
        >>>     transforms_cls.ToTensor()
        >>>     ])
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, bboxes, labels):
        for t in self.transforms:
            img, bboxes, labels = t(img, bboxes, labels)
        return img, bboxes, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lambda(object):
    """Apply a user-defined lambda as function.
    
    Args:
        lambd (function): Lambda/function to be used for transform.
    
    """
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, img, bboxes, labels):
        return self.lambd(img, bboxes, labels)

    def __repr__(self):
        return self.__class__.__namme + '()'


class ToTensor(object):
    """onvert numpy.ndarray to torch tensor.

        if the image is uint8 , it will be divided by 255;
        if the image is uint16 , it will be divided by 65535;
        if the image is float , it will not be divided, we suppose your image range should between [0~1] ;\n
    
    Args:
        img {numpy.ndarray} -- image to be converted to tensor.
        bboxes {numpy.ndarray} -- target bbox to be converted to tensor. the input should be [box_nums, 4]
        labels {numpy.ndarray} -- target labels to be converted to tensor. the input shape shold be [box_nums]
    """
    def __call__(self, img, bboxes, labels):

        return F.to_tensor(img), torch.tensor(bboxes, dtype=torch.float), torch.tensor(labels,dtype=torch.int)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.

    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        tensor (tensor): input torch tensor data.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace (boolean): inplace apply the transform or not. (default: False)
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor, bboxes, labels):
        return F.normalize(tensor, self.mean, self.std, self.inplace), bboxes, labels


class ToPILImage(object):
    # TODO
    pass


class ToGray(object):
    """Convert the image to grayscale
    
    Args:
        output_channels (int): number of channels desired for output image. (default: 1)
    
    Returns:
        [ndarray]: the graysacle version of input
        - If output_channels=1 : returned single channel image (height, width)
        - If output_channels>1 : returned multi-channels ndarray image (height, width, channels)
    """
    def __init__(self, output_channels=1):
        self.output_channels = output_channels
    def __call__(self, img, bboxes, labels):
        return F.to_grayscale(img, self.output_channels), bboxes, labels



class GaussianBlur(object):
    """Convert the input ndarray image to blurred image by gaussian method.
    
    Args:
        kernel_size (int): kernel size of gaussian blur method. (default: 3)
    
    Returns:
        ndarray: the blurred image.
    """
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img, bboxes, labels):
        return F.gaussian_blur(img, self.kernel_size), bboxes, labels


class RandomNoise(object):
    """Add noise to the input ndarray image.
    Args:
        mode (str): the noise mode, should be one of ``gaussian``, ``salt``, ``pepper``, ``s&p``, (default: gaussian).
        percent (float): noise percent, only work for ``salt``, ``pepper``, ``s&p`` mode. (default: 0.02)
    
    Returns:
        ndarray: noised ndarray image.
    """
    def __init__(self, mode='gaussian', percent=0.02):
        if mode not in ['gaussian', 'salt', 'pepper', 's&p']:
            raise ValueError('mode should be gaussian, salt, pepper, but got {}'.format(mode))
        self.mode = mode
        self.percent = percent
    def __call__(self, img, bboxes, labels):
        return F.noise(img, self.mode, self.percent), bboxes, labels

class RandomBrightness(object):
    def __init__(self, max_value=0):
        if isinstance(max_value, numbers.Number):
            self.value = random.uniform(-max_value, max_value)
        if isinstance(max_value, collections.Iterable) and len(max_value)==2:
            self.value = random.uniform(max_value[0], max_value[1])
    
    def __call__(self, img, bboxes, labels):
        return F.adjust_brightness(img, self.value), bboxes, labels


class RandomContrast(object):
    def __init__(self, max_factor=0):
        if isinstance(max_factor, numbers.Number):
            self.factor = random.uniform(-max_factor, max_factor)
        if isinstance(max_factor, collections.Iterable) and len(max_factor)==2:
            self.factor = random.uniform(max_factor[0], max_factor[1])
    
    def __call__(self, img, bboxes, labels):
        return F.adjust_contrast(img, self.factor), bboxes, labels

