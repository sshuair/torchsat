from functools import wraps
import torch
import numpy as np
import cv2
import collections
from PIL import Image
import numbers

__numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'uint16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

'''image functional utils

'''

# NOTE: all the function should recive the ndarray like image, should be W x H x C or W x H

# 如果将所有输出的维度够搞成height，width，channel 那么可以不用to_tensor??, 不行
def preserve_channel_dim(func):
    """Preserve dummy channel dim."""
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(img):
    '''convert numpy.ndarray to torch tensor. \n
        if the image is uint8 , it will be divided by 255;\n
        if the image is uint16 , it will be divided by 65535;\n
        if the image is float , it will not be divided, we suppose your image range should between [0~1] ;\n
    
    Arguments:
        img {numpy.ndarray} -- image to be converted to tensor.
    '''
    if not _is_numpy_image(img):
        raise TypeError('data should be numpy ndarray. but got {}'.format(type(img)))

    if img.ndim == 2:
        img = img[:, :, None]

    img = torch.from_numpy(img.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    elif isinstance(img, torch.ShortTensor):
        return img.float().div(65535)
    else:
        return img


def to_pil_image(tensor):
    # TODO
    pass


def to_tiff_image(tensor):
    # TODO
    pass


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor


def resize(img, size, interpolation=Image.BILINEAR):
    '''resize the image
    
    Arguments:
        img {ndarray} -- the input ndarray image
        size {int, iterable} -- the target size, if size is intger,  width and height will be resized to same \
                                otherwise, the size should be tuple (height, width) or list [height, width]
                                
    
    Keyword Arguments:
        interpolation {Image} -- the interpolation method (default: {Image.BILINEAR})
    
    Raises:
        TypeError -- img should be ndarray
        ValueError -- size should be intger or iterable vaiable and length should be 2.
    
    Returns:
        img -- resize ndarray image
    '''

    if not _is_numpy_image(img):
        raise TypeError('img shoud be ndarray image [w, h, c] or [w, h], but got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size)==2)):
        raise ValueError('size should be intger or iterable vaiable(length is 2), but got {}'.format(type(size)))

    if isinstance(size, int):
        height, width = (size, size)
    else:
        height, width = (size[0], size[1])

    return cv2.resize(img, (width, height), interpolation=interpolation)


def pad(img, padding, mode='reflect'):
    """Pad the given image.
    Args:
        padding : int, padding width
        mode: str or function. contain{‘constant’,‘edge’,‘linear_ramp’,‘maximum’,‘mean’
            , ‘median’, ‘minimum’, ‘reflect’,‘symmetric’,‘wrap’}
    Examples
        --------
        >>> Transformed_img = pad(img, (20,20,20,20), mode='reflect')
    """
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Iterable) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_bottom = pad_top = padding[1]
    if isinstance(padding, collections.Iterable) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    if len(img.shape) == 2:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
    if len(img.shape) == 3:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=mode)
    
    return img


def crop(img, top, left, height, width):
    '''crop image 
    
    Arguments:
        img {ndarray} -- image to be croped
        top {int} -- top size
        left {int} -- left size 
        height {int} -- croped height
        width {int} -- croped width
    '''
    if not _is_numpy_image(img):
        raise TypeError('the input image should be numpy ndarray with dimension 2 or 3.'
            'but got {}'.format(type(img))
        )
    
    if width<0 or height<0 or left <0 or height<0:
        raise ValueError('the input left, top, width, height should be greater than 0'
            'but got left={}, top={} width={} height={}'.format(left, top, width, height)
        )

    if (left+width) > img.width or (top+height) > img.height:
        raise ValueError('the input crop width and height should be small or \
         equal to image width and height. ')

    if img.shape == 2:
        return img[top:(top+height), left:(left+width)]
    elif img.shape == 3:
        return img[top:(top+height), left:(left+width), :]


def center_crop(img, output_size):
    '''crop image
    
    Arguments:
        img {ndarray} -- input image
        output_size {number or sequence} -- the output image size. if sequence, should be [h, w]
    
    Raises:
        ValueError -- the input image is large than original image.
    
    Returns:
        ndarray image -- return croped ndarray image.
    '''
    if len(img.shape) == 2:
        img_height, img_width = img.shape
    else:
        img_height, img_width, _ = img.shape

    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    if output_size[0] > img_height or output_size[1] > img_width:
        raise ValueError('the output_size should not greater than image size, but got {}'.format(output_size))
    
    target_height, target_width = output_size

    top = int(round((img_height - target_height)/2))
    left = int(round((img_width - target_width)/2))

    return crop(img, top, left, target_height, target_width)
    

def resized_crop(img, top, left, height, width, size, interpolation=Image.BILINEAR):

    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img

def hflip(img):
    return cv2.flip(img, 0)

def vflip(img):
    return cv2.flip(img, 1)

def five_crop():
    pass

def ten_crop():
    pass

def adjust_brightness():
    pass

def adjust_contrast():
    pass

def adjust_saturation():
    pass

def adjust_hue():
    pass

def adjust_gamma():
    pass

def rotate():
    pass

def affine():
    pass

def to_grayscale():
    pass