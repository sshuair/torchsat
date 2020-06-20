<p align="center">
  <img width="60%" height="60%" src="https://github.com/sshuair/torchsat/blob/master/docs/source/_static/img/logo-black.png">
</p>

--------------------------------------------------------------------------------
<p align="center">
    <a href="https://github.com/sshuair/torchsat/actions"><img src="https://github.com/sshuair/torchsat/workflows/pytest/badge.svg"></a>
    <a href="https://torchsat.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/torchsat/badge/?version=latest"></a>
    <a href="https://github.com/sshuair/torchsat/stargazer"><img src="https://img.shields.io/github/stars/sshuair/torchsat"></a>
    <a href="https://github.com/sshuair/torchsat/network"><img src="https://img.shields.io/github/forks/sshuair/torchsat"></a>
    <a href="https://github.com/sshuair/torchsat/blob/master/LICENSE"><img src="https://img.shields.io/github/license/sshuair/torchsat"></a>
</p>

TorchSat is an open-source deep learning framework for satellite imagery analysis based on PyTorch.

>This project is still work in progress. If you want to know the latest progress, please check the [develop](https://github.com/sshuair/torchsat/tree/develop) branch.

**Hightlight**
- :wink: Support multi-channels(> 3 channels, e.g. 8 channels) images and TIFF file as input.
- :yum: Convenient data augmentation method for classification, sementic segmentation and object detection.
- :heart_eyes: Lots of models for satellite vision tasks, such as ResNet, DenseNet, UNet, PSPNet, SSD, FasterRCNN ...
- :smiley: Lots of common satellite datasets loader.
- :open_mouth: Training script for common satellite vision tasks.

## Install
- source: `python3 setup.py install`

## How to use
- [Introduction](https://torchsat.readthedocs.io/en/latest/index.html) 
- Image Classification Tutorial: [Docs](https://torchsat.readthedocs.io/en/latest/tutorials/image-classification.html),  [Google Colab](https://colab.research.google.com/drive/1RLiz6ugYfR8hWP5vNkLjdyKjr6FY8SEy)
- Semantic Segmentation Tutorial: [Docs](https://torchsat.readthedocs.io/en/latest/tutorials/semantic-segmentation.html)
- Data Augumentation: [Docs](https://torchsat.readthedocs.io/en/latest/tutorials/data-augumentation.html), [Google Colab](https://colab.research.google.com/drive/1M46TXAM-JNV708Wn0OQDDXnD5nK9yUOK)


## Features

### Data augmentation

We suppose all the input images, masks and bbox should be NumPy ndarray. The data shape should be **[height, width]** or **[height, width, channels]**.

#### pixel level

Pixel-level transforms only change the input image and will leave any additional targets such as masks, bounding boxes unchanged. It support all channel images. Some transforms only support specific input channles.

| Transform            | Image  |  masks | BBoxes |
| -------------------- | :---:  |  :---: | :----: |
| ToTensor             |   ✓    |  ✓     |   ✓    |
| Normalize            |   ✓    |  ✓     |   ✓    |
| ToGray               |   ✓    |  ✓     |   ✓    |
| GaussianBlur         |   ✓    |  ✓     |   ✓    |
| RandomNoise          |   ✓    |  ✓     |   ✓    |
| RandomBrightness     |   ✓    |  ✓     |   ✓    |
| RandomContrast       |   ✓    |  ✓     |   ✓    |

#### spatial-level
Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes. It support all channel images.

| Transform            | Image | masks | BBoxes |
| -------------------- | :---: | :---: | :----: |
| Resize               |   ✓   |   ✓   |   ✓    |
| Pad                  |   ✓   |   ✓   |   ✓    |
| RandomHorizontalFlip |   ✓   |   ✓   |   ✓    |
| RandomVerticalFlip   |   ✓   |   ✓   |   ✓    |
| RandomFlip           |   ✓   |   ✓   |   ✓    |
| CenterCrop           |   ✓   |   ✓   |   ✓    |
| RandomCrop           |   ✓   |   ✓   |   ✓    |
| RandomResizedCrop    |   ✓   |   ✓   |   ✓    |
| ElasticTransform     |   ✓   |   ✓   |        |
| RandomRotation       |   ✓   |   ✓   |   ✓    |
| RandomShift          |   ✓   |   ✓   |   ✓    |


### Models
#### Classification
All models support multi-channels as input (e.g. 8 channels).
- VGG: `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19_bn`, `vgg19`
- ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`,`resnext101_32x8d`, `wide_resnet50_2`, `wide_resnet101_2`
- DenseNet: `densenet121`, `densenet169`, `densenet201`
- Inception: `inception_v3`
- MobileNet: `mobilenet_v2`
- EfficientNet: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`,`efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7`
- ResNeSt: `resnest50`, `resnest101`, `resnest200`, `resnest269`

#### Sementic Segmentation
- UNet: `unet`, `unet34`, `unet101`, `unet152` (with resnet as backbone.)


### Dataloader
#### Classification
- [SAT-4 and SAT-6 airborne datasets](https:/csc.lsu.edu/~saikat/deepsat/)
- [EuroSat](http:/madm.dfki.de/downloads)
- [PatternNet](https:/sites.google.com/view/zhouwx/dataset)
- [NWPU_redisc45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html#)


## Showcase
If you extend this repository or build projects that use it, we'd love to hear from you.


## Reference
- [torchvision](https://github.com/pytorch/vision)

## Note
- If you are looking for the torchvision-enhance, please checkout the [enhance](https://github.com/sshuair/torchvision-enhance/tree/torchvision-enhance) branch. But it was deprecated.
