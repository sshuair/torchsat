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

    if img.dtype == np.uint8:
        img = img.astype(np.float)/255
    elif img.dtype == np.uint16:
        img = img.astype(np.float)/65535
    elif img.dtype == np.float:
        img = img.astype(np.float)/1
    else:
        raise TypeError('{} is not support'.format(img.dtype))
    
    img = torch.from_numpy(img.transpose((2, 0, 1)))

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

def noise(img, mode='gaussain', percent=0.004):
    if mode == 'gaussian':
        mean = 0
        var = 0.1
        sigma = var*0.5
        if img.dim == 2:
            h, w = img.shape
            gauss = np.random.normal(mean, sigma, (h, w))
        else:
            h, w, c = img.shape
            gauss = np.random.normal(mean, sigma, (h, w, c))
        noisy = img + gauss

    elif mode == 'salt':
        row,col,ch = img.shape
        s_vs_p = 0.5
        num_salt = np.ceil(percent * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        img[coords] = 1
    
    elif mode == 'pepper':
        h, w, c = img.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        # Pepper mode
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in img.shape]
        out[coords] = 0

    elif mode == 's&p':
        row,col,ch = img.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in img.shape]
        out[coords] = 0

    return noisy


def gaussian_blur(img, kernel_size):
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)


def shift(img, top, left):
    (h, w) = img.shape[0:2]
    matrix = np.float32([[1, 0, left], [0, 1, top]])
    dst = cv2.warpAffine(img, matrix, (w, h))

    return dst
    

def rotate(img, angle, center=None, scale=1.0):
    (h, w) = img.shape[:2]
 
    if center is None:
        center = (w / 2, h / 2)
 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
 
    return rotated


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

    if img.ndim == 2:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
    if img.ndim == 3:
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

    if img.ndim == 2:
        return img[top:(top+height), left:(left+width)]
    elif img.ndim == 3:
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
    if img.ndim == 2:
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


def elastic_transform(image, alpha, sigma, alpha_affine, interpolation=cv2.INTER_LINEAR,
                      border_mode=cv2.BORDER_REFLECT_101, random_state=None, approximate=False):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(1234)

    height, width = image.shape[:2]

    # Random affine
    center_square = np.float32((height, width)) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, matrix, (width, height), flags=interpolation, borderMode=border_mode)

    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = (random_state.rand(height, width).astype(np.float32) * 2 - 1)
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha

        dy = (random_state.rand(height, width).astype(np.float32) * 2 - 1)
        cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
        dy *= alpha
    else:
        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)
        dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)

    return cv2.remap(image, mapx, mapy, interpolation, borderMode=border_mode)