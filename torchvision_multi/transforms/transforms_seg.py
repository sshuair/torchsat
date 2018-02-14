# coding:utf-8
"""transforms for semantic segmentation on input image and target(label) image.
"""
import random
from PIL import Image
from . import functional as F


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

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
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, it must be of len 2.")
        self.degrees = degrees

    def __call__(self, img, target):
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
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
        target = F.shift(img, right_shift=right_shift, down_shift=down_shift)
        
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

        top = random.randint(0, int(img.shape[0] - width + 1))
        left = random.randint(0, int(img.shape[1] - height + 1))

        if (width > img.shape[0] or height > img.shape[1]):
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


here = None

class SegRandomNoise(object):

    def __init__(self, probability, uintpara=16, mean=0, var=None):
        if not 0 <= probability <= 1:
            raise ValueError('SegRandomNoise.probability error')
        if not (uintpara == 8 or uintpara == 16):
            raise ValueError('SegRandomNoise.uintpara error')

        self.uintpara = uintpara
        self.probability = probability
        self.mean = mean
        self.var = var

    def __call__(self, img, target):
        r = round(random.uniform(0, 1), 1)
        if r < self.probability:
            img_trans = noise(img, self.uintpara, self.mean, self.var)
            return img_trans, target
        else:
            return img, target


class SegGaussianBlur(object):
    def __init__(self, probability, sigma=1, multichannel=True):
        if not 0 <= probability <= 1:
            raise ValueError('SegGaussianBlur.probability error')
        if sigma < 0:
            raise ValueError('SegGaussianBlur.sigma error')
        self.probability = probability
        self.sigma = sigma
        self.multichannel = multichannel

    def __call__(self, img, target):
        r = round(random.uniform(0, 1), 1)
        if r < self.probability:
            img_trans = gaussianblur(img, self.sigma, self.multichannel)
            return img_trans, target
        else:
            return img, target


class SegPieceTransfor(object):

    def __init__(self, probability, numcols=10, numrows=10, warp_left_right=10, warp_up_down=10):
        if not 0 <= probability <= 1:
            raise ValueError('SegPieceTransfor.probability error')
        if numcols < 0:
            raise ValueError('SegPieceTransfor.numcols error')
        if numrows < 0:
            raise ValueError('SegPieceTransfor.numrows error')
        if warp_left_right < 0:
            raise ValueError('SegPieceTransfor.warp_left_right error')
        if warp_up_down < 0:
            raise ValueError('SegPieceTransfor.warp_up_down error')

        self.probability = probability
        self.numcols = numcols
        self.numrows = numrows
        self.warp_left_right = warp_left_right
        self.warp_up_down = warp_up_down

    def __call__(self, img, target):
        r = round(random.uniform(0, 1), 1)
        if r < self.probability:
            img_trans = piecetransform(img, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down)
            target_trans = piecetransform(target, self.numcols, self.numrows, self.warp_left_right, self.warp_up_down,order=0)
            return img_trans,target_trans
        else:
            return img, target


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

        return to_tensor(img), torch.from_numpy(target).long()


# ======================================== common use class ============================================================

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img




class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


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
