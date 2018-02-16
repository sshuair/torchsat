from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
from skimage.external import tifffile
from skimage import transform, filters, util
import cv2


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # get max value of dtype, if the dtype is not uint8 and uint 16, change this
        # denominator = _get_dtype_max(pic)
        denominator = np.iinfo(np.uint8).max if pic.dtype == np.uint8 else np.iinfo(np.uint16).max

        # handle numpy array
        if len(pic.shape) == 2:
            img = torch.from_numpy(pic)
            img = torch.unsqueeze(img, 0)
        else:
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(denominator)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return


def to_ndarray(pic, dtype='uint8'):
    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        if dtype == 'uint8':
            npimg = (pic.numpy() * np.iinfo(np.uint8).max).astype(np.uint8)
        elif dtype == 'uint16':
            npimg = (pic.numpy() * np.iinfo(np.int32).max).astype(np.int32)
        else:
            raise ValueError('not support dtype')
        npimg = np.transpose(npimg, (1, 2, 0))
        
    else:
        npimg = np.transpose(pic.numpy(), (1, 2, 0))
    assert isinstance(npimg, np.ndarray)

    return npimg


def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    # TODO: make efficient
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def flip(img, flip_mode):
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not (isinstance(flip_mode, int)):
        raise TypeError('flipCode should be integer. Got {}'.format(type(flip_mode)))
    return cv2.flip(img, flip_mode)


def rotate(img, angle=0, order=1):
    """Rotate image by a certain angle around its center.

        Parameters
        ----------
        img : ndarray(uint16 or uint8)
            Input image.
        angle : integer
            Rotation angle in degrees in counter-clockwise direction.

        Returns
        -------
        rotated : ndarray(uint16 or uint8)
                Rotated version of the input.

        Examples
        --------
        rotate(image, 30)
        rotate(image, 180)
    """

    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not (isinstance(angle, numbers.Number)):
        raise TypeError('Angle should be integer. Got {}'.format(type(angle)))

    img_new = transform.rotate(img, angle, order=order, preserve_range=True)
    img_new = img_new.astype(img.dtype)

    return img_new


def shift(img, right_shift=5, down_shift=5):
    """

    :param img: the image input
    :param right_shift: the pixels of shift right
    :param down_shift: the pixels of down right
    :return: transformed img

    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not (isinstance(right_shift, int)):
        raise TypeError('shift.rightshift should be integer. Got {}'.format(type(right_shift)))
    if not (isinstance(down_shift, int)):
        raise TypeError('shift.downshift should be integer. Got {}'.format(type(down_shift)))

    tform = transform.SimilarityTransform(translation=(-right_shift, -down_shift))
    img_new = transform.warp(img, tform, preserve_range=True)
    img_new = img_new.astype(img.dtype)
    return img_new


def crop(img, top, left, width, height):
    """crop image from position top and left, width and height
    
    Arguments:
        img {numpy.ndarray} -- input image
        top {int} -- start position row
        left {int} -- start position column
        width {int} -- crop width
        height {int} -- crop height
    """
    if not all([isinstance(x, int) for x in (top, left, width, height)]):
        raise ValueError("params should be integer!")
    if (width > img.shape[0] or height > img.shape[1]):
        raise ValueError("the output imgage size should be small than input image!!!")

    if len(img.shape) == 2:
        img_height, img_width = img.shape
    else:
        img_height, img_width, _ = img.shape
    right = img_width - (left + width)
    bottom = img_height - (top + height)
    if len(img.shape) == 2:
        img_croped = util.crop(img,((top,bottom),(left,right)))
    else:
        img_croped = util.crop(img,((top,bottom),(left,right),(0,0)))

    return img_croped


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))

    return crop(img, i, j, th, tw)


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Note: cv2.resize do not support int32, weird
    Args:
        img (Numpy Array): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """

    if not _is_numpy_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        size = (size, size)
    
    if img.dtype == np.int32:
        resized = img.astype(np.uint16)
        resized = cv2.resize(resized, size , interpolation)
    else:
        resized = cv2.resize(img, size , interpolation)
    resized = resized.astype(img.dtype)
    return resized


def crop_resize(img, top, left, width, height, size, interpolation=Image.BILINEAR):
    """Crop the given PIL Image and resize it to desired size.

    Notably used in RandomResizedCrop.

    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be PIL Image'
    img = crop(img, top, left, width, height)
    img = resize(img, size, interpolation)
    return img


def pad(img, pad_width, mode='reflect', **kwargs):
    """Pad the given image.
    Args:
        pad_width : int, padding width
        mode: str or function. contain{‘constant’,‘edge’,‘linear_ramp’,‘maximum’,‘mean’
            , ‘median’, ‘minimum’, ‘reflect’,‘symmetric’,‘wrap’}
    Examples
        --------
        >>> Transformed_img = pad(img,[(20,20),(20,20),(0,0)],mode='reflect')
    """
    if len(img.shape) == 2:
        pad_width = ((pad_width, pad_width), (pad_width, pad_width))
    else:
        pad_width = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    
    return np.pad(img, pad_width, mode)


def noise(img, dtype='uint8', mode='gaussian', mean=0, var=0.01):
    """TODO
    """
    if dtype == 'uint16':
        img_new = img.astype(np.uint16)
                  
    img_new = util.random_noise(img, mode, mean=mean, var=var)

    if dtype == 'uint8':
        img_new = (img_new * np.iinfo(np.uint8).max).astype(np.uint8)
    elif dtype == 'uint16':
        img_new = (img_new * np.iinfo(np.int32).max).astype(np.int32)
    else:
        raise ValueError('not support type')

    return img_new


def gaussian_blur(img, sigma=1, dtype='uint8', multichannel=False):
    """Multi-dimensional Gaussian filter.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale or color) to filter.
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    multichannel : bool, optional (default: None)
        Whether the last axis of the image is to be interpreted as multiple
        channels. If True, each channel is filtered separately (channels are
        not mixed together). Only 3 channels are supported. If `None`,
        the function will attempt to guess this, and raise a warning if
        ambiguous, when the array has shape (M, N, 3).

    Returns
    -------
    filtered_image : ndarray
    """

    if np.any(np.asarray(sigma) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")
    img_new = filters.gaussian(img, sigma, multichannel)
    if dtype == 'uint8':
        print(img_new.max(), img_new.min())
        img_new = (img_new * np.iinfo(np.uint8).max).astype(np.uint8)
    elif dtype == 'uint16':
        img_new = (img_new * np.iinfo(np.int32).max).astype(np.int32)
    else:
        raise ValueError('not support type')
    return img_new


def piecewise_transform(image, numcols=5, numrows=5, warp_left_right=10, warp_up_down=10, order=1):
    """2D piecewise affine transformation.

        Control points are used to define the mapping. The transform is based on
        a Delaunay triangulation of the points to form a mesh. Each triangle is
        used to find a local affine transform.

        Parameters
        ----------
        img : ndarray
        numcols : int, optional (default: 5)
            numbers of the colums to transformation
        numrows : int, optional (default: 5)
            numbers of the rows to transformation
        warp_left_right: int, optional (default: 10)
            the pixels of transformation left and right
        warp_up_down: int, optional (default: 10)
            the pixels of transformation up and down
        Returns
        -------
        Transformed_image : ndarray

        Examples
        --------
            >>> Transformed_img = piecetransform(image,numcols=10, numrows=10, warp_left_right=5, warp_up_down=5)

        """

    rows, cols = image.shape[0], image.shape[1]

    numcols = numcols
    numrows = numrows

    src_cols = np.linspace(0, cols, numcols, dtype=int)
    src_rows = np.linspace(0, rows, numrows, dtype=int)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    src_rows_new = np.ndarray.transpose(src_rows)
    src_cols_new = np.ndarray.transpose(src_cols)
    # src_new = np.dstack([src_cols_new.flat, src_rows_new.flat])[0]

    dst_cols = np.ndarray(src_cols.shape)
    dst_rows = np.ndarray(src_rows.shape)
    for i in range(0, numcols):
        for j in range(0, numrows):
            if src_cols[i, j] == 0 or src_cols[i, j] == cols:
                dst_cols[i, j] = src_cols[i, j]
            else:
                dst_cols[i, j] = src_cols[i, j] + np.random.uniform(-1, 1) * warp_left_right

            if src_rows[i, j] == 0 or src_rows[i, j] == rows:
                dst_rows[i, j] = src_rows[i, j]
            else:
                dst_rows[i, j] = src_rows[i, j] + np.random.uniform(-1, 1) * warp_up_down

    dst = np.dstack([dst_cols.flat, dst_rows.flat])[0]

    # dst_rows_new = np.ndarray.transpose(dst_rows)
    # dst_cols_new = np.ndarray.transpose(dst_cols)
    # dst_new = np.dstack([dst_cols_new.flat, dst_rows_new.flat])[0]

    tform = transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    img_new = transform.warp(image, tform, output_shape=(rows, cols), order=order, preserve_range=True)
    img_new = img_new.astype(image.dtype)
    
    return img_new



if __name__ == '__main__':
    img = tifffile.imread('sample-data/MUL_AOI_4_Shanghai_img1920.tif')
    # tifffile.imshow(img[:,:,[3,2,1]])
    img = img.astype(np.int32)
    out = resize(img, (224,224))
    # out = filters.gaussian_filter(img, sigma=2)
    # out2 = gaussian_blur(img, dtype='uint16')
    # out2 = to_tensor(out2)
    # out3 = to_ndarray(out2, dtype='uint16')
    # pass