from . import functional as F
import random
import numbers

__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "Scale", "CenterCrop", "Pad",
           "Lambda", "RandomApply", "RandomChoice", "RandomOrder", "RandomCrop", "RandomHorizontalFlip",
           "RandomVerticalFlip", "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation",
           "AffineTransformation", "ColorJitter", "RandomRotation", "RandomAffine", "Grayscale", "RandomGrayscale"]


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

class ToPILImage(object):
    # TODO
    pass


class Normalize(object):

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std, self.inplace)


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
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w, _  = img.shape
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
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)