import numbers
import random

import cv2
from PIL import Image

from . import functional as F

__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "CenterCrop", "Pad",
           "Lambda", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", 
           "RandomResizedCrop", "RandomRotation",
           "AffineTransformation",  "RandomAffine", "Grayscale",
           'RandomShift', 'PieceTransform']


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lambda(object):
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__namme + '()'


class ToTensor(object):
    def __call__(self, img):

        return F.to_tensor(img)


class Normalize(object):

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std, self.inplace)


class ToPILImage(object):
    # TODO
    pass


class ToGray(object):
    def __init__(self, output_channels=1):
        self.output_channels = output_channels
    def __call__(self, img):
        return F.to_grayscale(img, self.output_channels)



class GaussianBlur(object):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        return F.gaussian_blur(img, self.kernel_size)



class RandomNoise(object):
    def __init__(self, mode='gaussian'):
        if mode not in ['gaussian', 'salt', 'pepper', 's&p']:
            raise ValueError('mode should be gaussian, salt, pepper, but got {}'.format(mode))
        self.mode=mode
    def __call__(self, img):
        return F.noise(img, self.mode)



class RandomShift(object):
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
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        return F.resize(img, self.size, self.interpolation)


class Pad(object):
    def __init__(self, padding, mode='reflect'):
        self.padding = padding
        self.mode = mode
    
    def __call__(self, img):
        return F.pad(img, self.padding, self.mode)


class CenterCrop(object):
    def __init__(self, out_size):
        self.out_size = out_size
    
    def __call__(self, img):
        return F.center_crop(img, self.out_size)


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[0:2]
        th, tw = size
        if w == tw and h == tw:
            return img

        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)

        return F.crop(img, top, left, th, tw)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return F.hflip(img)
        return img
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return F.vflip(img)
        return img
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(object):
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
        img = F.resize(self.target_size, interpolation=self.interpolation)



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

    def __init__(self, alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, random_state=None, approximate=False):
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.random_state = random_state
        self.approximate = approximate

    def __call__(self, img):
        return F.elastic_transform(img, self.alpha, self.sigma, self.alpha_affine, self.interpolation,
                                   self.border_mode, np.random.RandomState(self.random_state),
                                   self.approximate)

