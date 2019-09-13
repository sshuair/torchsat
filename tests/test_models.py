import pytest
import torch
# from torchsat.models.classification import resnet, densenet, vgg, inception, mobilenet
import torchsat.models.classification as clss
import torchsat.models.segmentation as seg

IN_CHANNELS = [1, 3, 8]

cls_models = [k for k, v in clss.__dict__.items() if callable(v) and k.lower()==k and k[0]!='_']

@pytest.mark.parametrize('model_name', cls_models)
def test_classification_models(model_name):
    assert clss.__dict__[model_name](2, in_channels=1,  pretrained=False)
    assert clss.__dict__[model_name](3, in_channels=2,  pretrained=False)
    assert clss.__dict__[model_name](4, in_channels=3,  pretrained=False)
    assert clss.__dict__[model_name](5, in_channels=7,  pretrained=False)
    assert clss.__dict__[model_name](6, in_channels=3,  pretrained=False)
    with pytest.raises(ValueError):
        clss.__dict__[model_name](2, in_channels=4,  pretrained=True)
    
    num_classes=10
    in_channels=5
    batch_size=2
    model = clss.__dict__[model_name](num_classes, in_channels=in_channels)
    model.eval()
    if model_name in ['inception_v3']:
        input_shape = (batch_size, in_channels, 299, 299)
    else:
        input_shape = (batch_size, in_channels, 256, 256)
    x = torch.rand(size=input_shape)
    out = model(x)
    assert out.shape[-1] == num_classes


seg_models = [k for k, v in seg.__dict__.items() if callable(v) and k.lower()==k and k[0]!='_']
@pytest.mark.parametrize('model_name', seg_models)
def test_segmentation_models(model_name):
    assert seg.__dict__[model_name](2, in_channels=1,  pretrained=False)
    assert seg.__dict__[model_name](3, in_channels=2,  pretrained=False)
    assert seg.__dict__[model_name](4, in_channels=3,  pretrained=False)
    assert seg.__dict__[model_name](5, in_channels=7,  pretrained=False)
    assert seg.__dict__[model_name](6, in_channels=3,  pretrained=False)
    with pytest.raises(ValueError):
        seg.__dict__[model_name](2, in_channels=4,  pretrained=True)
    
    num_classes=10
    in_channels=5
    batch_size=2
    model = seg.__dict__[model_name](num_classes, in_channels=in_channels)
    model.eval()
    input_shape = (batch_size, in_channels, 256, 256)
    x = torch.rand(size=input_shape)
    out = model(x)
    assert out.shape[0] == batch_size
    assert list(out.shape[2:]) == [input_shape[-2], input_shape[-1]]


def test_temp():
    clss.resnet18(2, in_channels=3, pretrained=False)