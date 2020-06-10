import collections
import numbers
import random

from PIL import Image
import numpy as np
import cv2
import torch

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
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pre_img, post_img, mask):
        for t in self.transforms:
            pre_img, post_img, mask = t(pre_img, post_img, mask)
        return pre_img, post_img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Lambda(object):
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, pre_img, post_img, mask):
        return self.lambd(pre_img, post_img, mask)

    def __repr__(self):
        return self.__class__.__namme + "()"


class ToTensor(object):
    def __call__(self, pre_img, post_img, mask):
        return F.to_tensor(pre_img), F.to_tensor(post_img), torch.tensor(mask, dtype=torch.long)


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, pre_tensor, post_tensor, mask):
        return F.normalize(pre_tensor, self.mean, self.std, self.inplace), \
               F.normalize(post_tensor, self.mean, self.std, self.inplace), \
               mask


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

    def __call__(self, img, mask):
        return F.to_grayscale(img, self.output_channels), mask


class RandomNoise(object):
    def __init__(self, mode="gaussian"):
        if mode not in ["gaussian", "salt", "pepper"]:
            raise ValueError(
                "mode should be gaussian, salt, pepper, but got {}".format(mode)
            )

    def __call__(self, pre_img, post_img, mask):
        return F.noise(pre_img, self.mode), F.noise(post_img, self.mode), mask


class GaussianBlur(object):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, pre_img, post_img, mask):
        return F.gaussian_blur(pre_img, self.kernel_size), F.gaussian_blur(post_img, self.kernel_size), mask


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

    def __call__(self, pre_img, post_img, mask):
        return F.noise(pre_img, self.mode, self.percent), F.noise(post_img, self.mode, self.percent), mask


class RandomBrightness(object):
    def __init__(self, max_value=0):
        if isinstance(max_value, numbers.Number):
            self.value = random.uniform(-max_value, max_value)
        if isinstance(max_value, collections.Iterable) and len(max_value) == 2:
            self.value = random.uniform(max_value[0], max_value[1])

    def __call__(self, pre_img, post_img, mask):
        return F.adjust_brightness(pre_img, self.value), F.adjust_brightness(pre_img, self.value), mask


class RandomContrast(object):
    def __init__(self, max_factor=0):
        if isinstance(max_factor, numbers.Number):
            self.factor = random.uniform(-max_factor, max_factor)
        if isinstance(max_factor, collections.Iterable) and len(max_factor) == 2:
            self.factor = random.uniform(max_factor[0], max_factor[1])

    def __call__(self, pre_img, post_img, mask):
        return F.adjust_contrast(pre_img, self.factor), F.adjust_contrast(post_img, self.factor), mask


class RandomShift(object):
    """random shift the ndarray with value or some percent.

    Args:
        max_percent (float): shift percent of the image.

    Returns:
        ndarray: return the shifted ndarray image.
    """

    def __init__(self, max_percent=0.4):
        self.max_percent = max_percent

    def __call__(self, pre_img, post_img, mask):
        height, width = img.shape[0:2]
        max_top = int(height * self.max_percent)
        max_left = int(width * self.max_percent)
        top = random.randint(-max_top, max_top)
        left = random.randint(-max_left, max_left)

        return F.shift(pre_img, top, left), F.shift(post_img, top, left), F.shift(mask, top, left)


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

    def __call__(self, pre_img, post_img, mask):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return F.rotate(pre_img, angle, self.center), \
               F.rotate(post_img, angle, self.center), \
               F.rotate(mask, angle, self.center)


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

    def __call__(self, pre_img, post_img, mask):
        return F.resize(pre_img, self.size, self.interpolation), \
            F.resize(post_img, self.size, self.interpolation), \
            F.resize(mask, self.size, Image.NEAREST),


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

    def __call__(self, pre_img, post_img, mask):
        pre_img = F.pad(pre_img, self.padding, self.fill, self.padding_mode)
        post_img = F.pad(post_img, self.padding, self.fill, self.padding_mode)
        if self.padding_mode == "reflect":
            return pre_img, post_img, F.pad(mask, self.padding, 0, self.padding_mode)
        else:
            return pre_img, post_img, F.pad(mask, self.padding, 0, "constant")


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

    def __call__(self, pre_img, post_img, mask):
        return F.center_crop(pre_img, self.out_size), \
               F.center_crop(post_img, self.out_size), \
               F.center_crop(mask, self.out_size)


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

    def __call__(self, pre_img, post_img, mask):
        h, w = pre_img.shape[0:2]
        th, tw = self.size
        if w == tw and h == tw:
            return pre_img, post_img

        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)

        return F.crop(pre_img, top, left, th, tw), F.crop(post_img, top, left, th, tw), F.crop(mask, top, left, th, tw)


class RandomHorizontalFlip(object):
    """Flip the input image on central horizon line.

    Args:
        p (float): probability apply the horizon flip.(default: 0.5)

    Returns:
        ndarray: return the flipped image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pre_img, post_img, mask):
        if random.random() < self.p:
            return F.hflip(pre_img), F.hflip(post_img), F.hflip(mask)
        return pre_img, post_img, mask

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

    def __call__(self, pre_img, post_img, mask):
        if random.random() < self.p:
            return F.vflip(pre_img), F.vflip(post_img), F.vflip(mask)
        return pre_img, post_img, mask

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

    def __call__(self, pre_img, post_img, mask):
        if random.random() < self.p:
            flip_code = random.randint(0, 1)
            return F.flip(pre_img, flip_code), F.flip(post_img, flip_code), F.flip(mask, flip_code)
        return pre_img, post_img, mask

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

    def __call__(self, pre_img, post_img, mask):
        h, w = pre_img.shape[0:2]
        th, tw = self.crop_size
        if w == tw and h == tw:
            return pre_img, post_img, mask

        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)

        pre_img = F.crop(pre_img, top, left, th, tw)
        pre_img = F.resize(pre_img, self.target_size, interpolation=self.interpolation)
        post_img = F.crop(post_img, top, left, th, tw)
        post_img = F.resize(post_img, self.target_size, interpolation=self.interpolation)
        mask = F.crop(mask, top, left, th, tw)
        mask = F.resize(mask, self.target_size, interpolation=Image.NEAREST)

        return pre_img, post_img, mask


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

    def __call__(self, pre_img, post_img, mask):
        return (
            F.elastic_transform(
                pre_img,
                self.alpha,
                self.sigma,
                self.alpha_affine,
                self.interpolation,
                self.border_mode,
                np.random.RandomState(self.random_state),
                self.approximate,
            ),
            F.elastic_transform(
                post_img,
                self.alpha,
                self.sigma,
                self.alpha_affine,
                self.interpolation,
                self.border_mode,
                np.random.RandomState(self.random_state),
                self.approximate,
            ),
            F.elastic_transform(
                mask,
                self.alpha,
                self.sigma,
                self.alpha_affine,
                cv2.INTER_NEAREST,
                self.border_mode,
                np.random.RandomState(self.random_state),
                self.approximate,
            ),
        )

