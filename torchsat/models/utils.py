
from .classification.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19
from .classification.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2,wide_resnet101_2 
from .classification.densenet import densenet121, densenet169, densenet201
from .classification.inception import inception_v3
from .classification.mobilenet import mobilenet_v2
from .segmentation.unet import unet34, unet101, unet152


__all__ = ["get_model"]

models = {
    'vgg11': vgg11, 
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
    'vgg19': vgg19,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
    'mobilenet_v2': mobilenet_v2,
    'inception_v3': inception_v3,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'unet34': unet34,
    'unet101': unet101,
    'unet152': unet152,
}


def get_model(name: str, num_classes: int, **kwargs):
    print(kwargs)
    if name.lower() not in models:
        raise ValueError("no model named {}, should be one of {}".format(name, ' '.join(models)))

    return models.get(name.lower())(num_classes, **kwargs)
