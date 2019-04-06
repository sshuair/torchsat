from . import functinal as F


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


# TODO & NOTE: 此处考虑图像分类、目标检测（coco、voc）、实例分割（cooc）三种情况
# 比如
# ToTensor：图像变，目标不变
# rotate：图像、目标都要变
# brightness：图像变， 目标不变

class ToTensor(object):
    def __call__(self, img, target):

        return F.to_tensor(img)
