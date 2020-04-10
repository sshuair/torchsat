import collections
import numbers
import random

import cv2
import numpy as np
from PIL import Image

from . import functional as F

__all__ = [
    "Compose",
    "Lambda",
    "ToTensor",
    "Normalize",
    "ToGray",
    "GaussianBlur",
    "RandomNoise",
    "RandomBrightness",
    "RandomContrast",
    "RandomShift",
    "RandomRotation",
    "Resize",
    "Pad",
    "CenterCrop",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomFlip",
    "RandomResizedCrop",
    "ElasticTransform",
]


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

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Lambda(object):
    """Apply a user-defined lambda as function.
    
    Args:
        lambd (function): Lambda/function to be used for transform.
    
    """

    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__namme + "()"


class ToTensor(object):
    """onvert numpy.ndarray to torch tensor.

        if the image is uint8 , it will be divided by 255;
        if the image is uint16 , it will be divided by 65535;
        if the image is float , it will not be divided, we suppose your image range should between [0~1] ;\n
    
    Args:
        img {numpy.ndarray} -- image to be converted to tensor.
    """

    def __call__(self, img):

        return F.to_tensor(img)


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

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std, self.inplace)


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

    def __call__(self, img):
        return F.to_grayscale(img, self.output_channels)


class GaussianBlur(object):
    """Convert the input ndarray image to blurred image by gaussian method.
    
    Args:
        kernel_size (int): kernel size of gaussian blur method. (default: 3)
    
    Returns:
        ndarray: the blurred image.
    """

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        return F.gaussian_blur(img, self.kernel_size)


class RandomNoise(object):
    """Add noise to the input ndarray image.
    Args:
        mode (str): the noise mode, should be one of ``gaussian``, ``salt``, ``pepper``, ``s&p``, (default: gaussian).
        percent (float): noise percent, only work for ``salt``, ``pepper``, ``s&p`` mode. (default: 0.02)
    
    Returns:
        ndarray: noised ndarray image.
    """

    def __init__(self, mode="gaussian", percent=0.02):
        if mode not in ["gaussian", "salt", "pepper", "s&p"]:
            raise ValueError(
                "mode should be gaussian, salt, pepper, but got {}".format(mode)
            )
        self.mode = mode
        self.percent = percent

    def __call__(self, img):
        return F.noise(img, self.mode, self.percent)


class RandomBrightness(object):
    def __init__(self, max_value=0):
        if isinstance(max_value, numbers.Number):
            self.value = random.uniform(-max_value, max_value)
        if isinstance(max_value, collections.Iterable) and len(max_value) == 2:
            self.value = random.uniform(max_value[0], max_value[1])

    def __call__(self, img):
        return F.adjust_brightness(img, self.value)


class RandomContrast(object):
    def __init__(self, max_factor=0):
        if isinstance(max_factor, numbers.Number):
            self.factor = random.uniform(-max_factor, max_factor)
        if isinstance(max_factor, collections.Iterable) and len(max_factor) == 2:
            self.factor = random.uniform(max_factor[0], max_factor[1])

    def __call__(self, img):
        return F.adjust_contrast(img, self.factor)


class RandomShift(object):
    """random shift the ndarray with value or some percent.
    
    Args:
        max_percent (float): shift percent of the image.
    
    Returns:
        ndarray: return the shifted ndarray image.
    """

    def __init__(self, max_percent=0.4):
        self.max_percent = max_percent

    def __call__(self, img):
        height, width = img.shape[0:2]
        max_top = int(height * self.max_percent)
        max_left = int(width * self.max_percent)
        top = random.randint(-max_top, max_top)
        left = random.randint(-max_left, max_left)

        return F.shift(img, top, left)


class RandomRotation(object):
    """random rotate the ndarray image with the degrees.
    
    Args:
        degrees (number or sequence): the rotate degree.
                                  If single number, it must be positive.
                                  if squeence, it's length must 2 and first number should small than the second one.
    
    Raises:
        ValueError: If degrees is a single number, it must be positive.
        ValueError: If degrees is a sequence, it must be of len 2.
    
    Returns:
        ndarray: return rotated ndarray image.
    """

    def __init__(self, degrees, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.center = center

    def __call__(self, img):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return F.rotate(img, angle, self.center)


class Resize(object):
    """resize the image
    Args:
        img {ndarray} : the input ndarray image
        size {int, iterable} : the target size, if size is intger,  width and height will be resized to same \
                                otherwise, the size should be tuple (height, width) or list [height, width]
                                
    
    Keyword Arguments:
        interpolation {Image} : the interpolation method (default: {Image.BILINEAR})
    
    Raises:
        TypeError : img should be ndarray
        ValueError : size should be intger or iterable vaiable and length should be 2.
    
    Returns:
        img (ndarray) : resize ndarray image
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return F.resize(img, self.size, self.interpolation)


class Pad(object):
    """Pad the given ndarray image with padding width.
    Args:
        padding : {int, sequence}, padding width 
                  If int, each border same.
                  If sequence length is 2, this is the padding for left/right and top/bottom.
                  If sequence length is 4, this is the padding for left, top, right, bottom.
        fill: {int, sequence}: Pixel
        padding_mode: str or function. contain{‘constant’,‘edge’,‘linear_ramp’,‘maximum’,‘mean’
            , ‘median’, ‘minimum’, ‘reflect’,‘symmetric’,‘wrap’} (default: constant)
    Examples:
        >>> transformed_img = Pad(img, 20, mode='reflect')
        >>> transformed_img = Pad(img, (10,20), mode='edge')
        >>> transformed_img = Pad(img, (10,20,30,40), mode='reflect')
    """

    def __init__(self, padding, fill=0, padding_mode="constant"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        return F.pad(img, self.padding, self.fill, self.padding_mode)


class CenterCrop(object):
    """crop image
    
    Args:
        img {ndarray}: input image
        output_size {number or sequence}: the output image size. if sequence, should be [height, width]
    
    Raises:
        ValueError: the input image is large than original image.
    
    Returns:
        ndarray: return croped ndarray image.
    """

    def __init__(self, out_size):
        self.out_size = out_size

    def __call__(self, img):
        return F.center_crop(img, self.out_size)


class RandomCrop(object):
    """random crop the input ndarray image
    
    Args:
        size (int, sequence): th output image size, if sequeue size should be [height, width]
    
    Returns:
        ndarray:  return random croped ndarray image.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[0:2]
        th, tw = self.size
        if w == tw and h == tw:
            return img

        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)

        return F.crop(img, top, left, th, tw)


class RandomHorizontalFlip(object):
    """Flip the input image on central horizon line.
    
    Args:
        p (float): probability apply the horizon flip.(default: 0.5)
    
    Returns:
        ndarray: return the flipped image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomVerticalFlip(object):
    """Flip the input image on central vertical line.
    
    Args:
        p (float): probability apply the vertical flip. (default: 0.5)
    
    Returns:
        ndarray: return the flipped image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomFlip(object):
    """Flip the input image vertical or horizon.
    
    Args:
        p (float): probability apply flip. (default: 0.5)
    
    Returns:
        ndarray: return the flipped image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            flip_code = random.randint(0, 1)
            return F.flip(img, flip_code)
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomResizedCrop(object):
    """[summary]
    
    Args:
        object ([type]): [description]
    
    Returns:
        [type]: [description]
    """

    def __init__(self, crop_size, target_size, interpolation=Image.BILINEAR):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img):
        h, w = img.shape[0:2]
        th, tw = self.crop_size
        if w == tw and h == tw:
            return img

        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)

        img = F.crop(img, top, left, th, tw)
        img = F.resize(img, self.target_size, interpolation=self.interpolation)

        return img


class ElasticTransform(object):
    """
    code modify from https://github.com/albu/albumentations.  
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    Args:
        approximate (boolean): Whether to smooth displacement map with fixed kernel size.
                               Enabling this option gives ~2X speedup on large images.
    Image types:
        uint8, uint16 float32
    """

    def __init__(
        self,
        alpha=1,
        sigma=50,
        alpha_affine=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        random_state=None,
        approximate=False,
    ):
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.random_state = random_state
        self.approximate = approximate

    def __call__(self, img):
        return F.elastic_transform(
            img,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            self.interpolation,
            self.border_mode,
            np.random.RandomState(self.random_state),
            self.approximate,
        )

