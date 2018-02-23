# coding:utf-8
"""transforms for semantic segmentation on input image and target(label) image.
"""
import random
import numbers
from PIL import Image
import torch
from . import functional as F

class SegCompose(object):
    """Composes several transforms together for segmantation.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class SegLambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, target):
        return self.lambd(img), target

    
class SegToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor for segmantation.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img, target):
        """
        Args:
            img (PIL.Image or numpy.ndarray): Image to be converted to tensor.
            target (numpy.ndarray): convert target to long tensor

        Returns:
            Tensor: Converted image.
        """

        return F.to_tensor(img), torch.from_numpy(target).long()


class SegNormalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, target):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        return F.normalize(tensor, self.mean, self.std), target


class SegVFlip(object):
    """
    random vertical flip the image
    """
    def __call__(self, img, target):
        if random.random() > 0.5:
            return F.flip(img, 1),  F.flip(target, 1),
        else:
            return img, target


class SegHFlip(object):
    """
    random horizontal flip the image
    """
    def __call__(self, img, target):
        if random.random() > 0.5:
            return F.flip(img, 0), F.flip(target, 0)
        else:
            return img, target


class SegRandomFlip(object):
    """
    random flip the ndarray image
    random probability (default init):
        1. original: 0.25
        2. left to right: 0.25
        3. top to bottom: 0.25
        4. left to right and then top to bottom: 0.25
    """
    def __call__(self, img, target):
        # if target==None:
        #     raise ValueError('SegRandomFlip has no target parameters ')
        flip_mode = random.randint(-1, 2)
        if flip_mode == 2:
            return img, target
        else:
            return F.flip(img, flip_mode), F.flip(target, flip_mode)


class SegRandomRotate(object):
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def __call__(self, img, target):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return F.rotate(img, angle), F.rotate(target, angle, order=0)


class SegRandomShift(object):
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
        
    def __call__(self, img, target):
        right_shift = random.randint(self.shift_dist[0], self.shift_dist[1])
        down_shift = random.randint(self.shift_dist[0], self.shift_dist[1])

        img = F.shift(img, right_shift=right_shift, down_shift=down_shift)
        target = F.shift(target, right_shift=right_shift, down_shift=down_shift)
        
        return img, target


class SegRandomCrop(object):
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

    def __call__(self, img, target):
        width, height = self.crop_size

        left = random.randint(0, int(img.shape[1] - width))
        top = random.randint(0, int(img.shape[0] - height))

        if (width > img.shape[1] or height > img.shape[0]):
            raise ValueError("the output imgage size should be small than input image!!!")

        return F.crop(img, top, left, width, height), F.crop(target, top, left, width, height)



class SegCenterCrop(object):
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

    def __call__(self, img, target):
        """
        Args:
            img (numpy array): Image to be cropped.

        Returns:
            Numpy Array: Cropped image.
        """
        return F.center_crop(img, self.size), F.center_crop(target, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class SegResize(object):
    """resize the image
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        
    def __call__(self, img, target):
        return F.resize(img, self.size), F.resize(target, self.size, Image.NEAREST)


class SegPad(object):
    
    def __init__(self, pad_width, mode='reflect'):
        if not isinstance(pad_width, int):
            raise ValueError("pad width should be intger, but got {}".format(type(pad_width)))
        self.pad_width = pad_width
        self.mode = mode

    def __call__(self, img, target):
        return F.pad(img, self.pad_width, self.mode), F.pad(target, self.pad_width, self.mode), 


class SegRandomNoise(object):
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

    def __call__(self, img, target):
        return F.noise(img, dtype=self.dtype, mode=self.mode, var=self.var), target


class SegGaussianBlur(object):
    def __init__(self, sigma=1, dtype='uint8', multichannel=True):
        if sigma<0:
            raise ValueError('GaussianBlur.sigma error')
        self.sigma = sigma
        self.dtype = dtype
        self.multichannel=multichannel

    def __call__(self, img, target):
        return F.gaussian_blur(img, self.sigma, self.dtype, self.multichannel), target


class SegPieceTransfor(object):
    def __init__(self, numcols=10, numrows=10, warp_left_right=10, warp_up_down=10):
        if numcols < 0:
            raise ValueError('SegPieceTransfor.numcols error')
        if numrows < 0:
            raise ValueError('SegPieceTransfor.numrows error')
        if warp_left_right < 0:
            raise ValueError('SegPieceTransfor.warp_left_right error')
        if warp_up_down < 0:
            raise ValueError('SegPieceTransfor.warp_up_down error')

        self.numcols = numcols
        self.numrows = numrows
        self.warp_left_right = warp_left_right
        self.warp_up_down = warp_up_down

    def __call__(self, img, target):
        img_trans = F.piecewise_transform(img, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down)
        target_trans = F.piecewise_transform(target, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down,order=0)
        return img_trans,target_trans


class SegPieceTransform(object):
    def __init__(self, numcols=5, numrows=5, warp_left_right=10, warp_up_down=10):
        if numcols<0 :
            raise ValueError('PieceTransfor.numcols error')
        if numrows<0 :
            raise ValueError('PieceTransfor.numrows error')
        if warp_left_right<0 :
            raise ValueError('PieceTransfor.warp_left_right error')
        if warp_up_down<0 :
            raise ValueError('PieceTransfor.warp_up_down error')

        self.numcols = numcols
        self.numrows = numrows
        self.warp_left_right = warp_left_right
        self.warp_up_down = warp_up_down

    def __call__(self, img, target):
        img_transform = F.piecewise_transform(img, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down)
        target_transform = F.piecewise_transform(target, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down)
        return img_transform, target_transform
