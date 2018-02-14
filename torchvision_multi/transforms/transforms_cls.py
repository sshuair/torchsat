#coding:utf-8
"""transform for classification
"""
import random
import numpy as np
import numbers
import collections

from . import functional as F


class Flip(object):
    """
    flip the input ndarray image
    three case:
        1. left to right
        2. top to bottom
        3. left to right and then top to bottom
    """
    def __init__(self, flip_mode):
        assert isinstance(flip_mode, int)
        self.flip_mode = flip_mode
    
    def __call__(self, img):
        return F.flip(img, self.flip_mode)


class VFlip(object):
    """
    random vertical flip the image
    """
    def __call__(self, img):
        if random.random() > 0.5:
            return F.flip(img, 1)
        else:
            return img


class HFlip(object):
    """
    random horizontal flip the image
    """
    def __call__(self, img):
        if random.random() > 0.5:
            return F.flip(img, 0)
        else:
            return img


class RandomFlip(object):
    """
    random flip the ndarray image
    random probability (default init):
        1. original: 0.25
        2. left to right: 0.25
        3. top to bottom: 0.25
        4. left to right and then top to bottom: 0.25
    """

    # def __init__(self, u=0.25):
    #     self.u = 0.25
    def __call__(self, img):
        flip_mode = random.randint(-1, 2)
        if flip_mode == 2:
            return img
        else:
            return F.flip(img, flip_mode)


class RandomRotate(object):
    """rotate the image.

        Args:
            degrees: tuple , min and max value of rotate value
        Example:
            >>> transform_multi.Randomrotate()

    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return F.rotate(img, angle)


class RandomShift(object):
    """shift the image.

        Args:
            probability:the probability of the operation (0<double<=1)
            rightshift: the scale of the shift right (pixels)
            downshift:the scale of the shift down (pixels)
        Example:
            >>> transform_multi.RandomShift(probability=1, rightshift=10, downshift=10)

    """
    def __init__(self, shift_dist):
        if isinstance(shift_dist, int):
            if shift_dist < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.shift_dist = (-shift_dist, shift_dist)
        else:
            if len(shift_dist) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            if not all([isinstance(x, int) for x in shift_dist]):
                raise ValueError("degrees must be positive.")
            self.shift_dist = shift_dist
        
    def __call__(self, img):
        right_shift = random.randint(self.shift_dist[0], self.shift_dist[1])
        down_shift = random.randint(self.shift_dist[0], self.shift_dist[1])
        return F.shift(img, right_shift=right_shift, down_shift=down_shift)


class RandomCrop(object):
    """shift the image.

        Args:
            outsize: the shape of the croped image
        Example:
            >>> transform_multi.RandomCrop(1, (256,256))

    """
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            if crop_size < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.crop_size = (crop_size, crop_size)
        else:
            if len(crop_size) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            if not all([isinstance(x, int) for x in crop_size]):
                raise ValueError("degrees must be positive.")
            self.crop_size = crop_size

    def __call__(self, img):
        width, height = self.crop_size

        top = random.randint(0, int(img.shape[0] - width + 1))
        left = random.randint(0, int(img.shape[1] - height + 1))

        if (width > img.shape[0] or height > img.shape[1]):
            raise ValueError("the output imgage size should be small than input image!!!")

        return F.crop(img, top, left, width, height)



class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be cropped.

        Returns:
            Numpy Array: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Resize(object):
    """resize the image
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        
    def __call__(self, img):
        return F.center_crop(img, self.size)


class Pad(object):
    
    def __init__(self, pad_width, mode='reflect'):
        if not isinstance(pad_width, int):
            raise ValueError("pad width should be intger, but got {}".format(type(pad_width)))
        self.pad_width = pad_width
        self.mode = mode

    def __call__(self, img):
        return F.pad(img, self.pad_width, self.mode)


class Noise(object):
    """Noise
    if dtype is uint8,  var should be 0.01 is best.
    if dtype is uint16, var should be 0.001 is best
    """

    def __init__(self, dtype='uint8', mode='gaussian', var=0.01):
        if dtype not in ('uint8', 'uint16'):
            raise ValueError('not support data type')
        self.dtype = dtype
        self.mode = mode
        self.var = var

    def __call__(self, img):
        return F.noise(img, dtype=self.dtype, mode=self.mode, var=self.var)


class GaussianBlur(object):
    def __init__(self, sigma=1, multichannel=True):
        if sigma<0:
            raise ValueError('GaussianBlur.sigma error')
        self.sigma = sigma
        self.multichannel=multichannel

    def __call__(self, img):
        return F.gaussianblur(img, self.sigma, self.multichannel)



class PieceTransform(object):
    def __init__(self, probability, numcols=10, numrows=10, warp_left_right=10, warp_up_down=10):
        if not 0<= probability <=1 :
            raise ValueError('PieceTransfor.probability error')
        if numcols<0 :
            raise ValueError('PieceTransfor.numcols error')
        if numrows<0 :
            raise ValueError('PieceTransfor.numrows error')
        if warp_left_right<0 :
            raise ValueError('PieceTransfor.warp_left_right error')
        if warp_up_down<0 :
            raise ValueError('PieceTransfor.warp_up_down error')

        self.probability = probability
        self.numcols = numcols
        self.numrows = numrows
        self.warp_left_right = warp_left_right
        self.warp_up_down = warp_up_down

    def __call__(self, img):
        r = round(random.uniform(0, 1), 1)
        # print(r, self.probability)
        if r < self.probability:
            return F.piecetransform(img, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down)
        else:
            return img


