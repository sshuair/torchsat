import pytest
from torchsat.models.classification import resnet, densenet, vgg, alexnet

IN_CHANNELS = [1, 3, 8]


def test_alexnet():
    assert alexnet.alexnet(pretrained=False, in_channels=1)
    assert alexnet.alexnet(pretrained=False, in_channels=3)
    assert alexnet.alexnet(pretrained=False, in_channels=4)
    assert alexnet.alexnet(pretrained=False, in_channels=8)
    assert alexnet.alexnet(pretrained=True, in_channels=3)
    with pytest.raises(ValueError): 
        alexnet.alexnet(pretrained=True, in_channels=5)


def test_vgg():
    assert vgg.vgg11(pretrained=False, in_channels=1)
    assert vgg.vgg11(pretrained=False, in_channels=3)
    assert vgg.vgg11(pretrained=False, in_channels=4)
    assert vgg.vgg11(pretrained=False, in_channels=8)
    assert vgg.vgg11(pretrained=True, in_channels=3)
    with pytest.raises(ValueError): 
        vgg.vgg11(pretrained=True, in_channels=5)


def test_resnet():
    assert resnet.resnet18(pretrained=False, in_channels=1, num_classes=10)
    assert resnet.resnet18(pretrained=False, in_channels=3)
    assert resnet.resnet18(pretrained=False, in_channels=4)
    assert resnet.resnet18(pretrained=False, in_channels=8)
    assert resnet.resnet18(pretrained=True, in_channels=3)
    with pytest.raises(ValueError): 
        resnet.resnet18(pretrained=True, in_channels=5)


def test_densenet():
    assert densenet.densenet121(pretrained=False, in_channels=1)
    assert densenet.densenet121(pretrained=False, in_channels=3)
    assert densenet.densenet121(pretrained=False, in_channels=8)
    assert densenet.densenet121(pretrained=True, in_channels=3)

    with pytest.raises(ValueError): 
        densenet.densenet121(pretrained=True, in_channels=5)