# :fire: TorchSat :earth_asia: 

TorchSat is an open-source deep learning framework for satellite imagery analysis based on PyTorch.

>This project is still work in progress. If you want to know more about it, please refer to the [Roadmap](https://github.com/sshuair/torchsat/wiki/Roadmap)

**Hightlight**
- :wink: Support multi-channels(> 3 channels, e.g. 8 channels) images and TIFF file as input.
- :yum: Convenient data augmentation method for classification, sementic segmentation and object detection.
- :heart_eyes: Lots of models for satellite vision tasks, such as ResNet, DenseNet, UNet, PSPNet, SSD, FasterRCNN ...
- :smiley: Lots of common satellite datasets loader.
- :open_mouth: Training script for common satellite vision tasks.

## Install

`python3 setup.py install`

## How to use
- **Introduction** - 
- **Classification tutorial** - 
- **Data augmentation** - [data-augmentation.ipynb](exsamples/data-augmentation.ipynb)
- **Data loader** 
- **models** 
- **train script** 

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
| RandomRotation       |   ✓   |   ✓   |        |
| RandomShift          |   ✓   |   ✓   |        |


### Models
#### Classification
All models support multi-channels as input (e.g. 8 channels).
- VGG: `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`,`vgg19_bn`, `vgg19`
- ResNet: `resnet18`, `resnet34`, `restnet50`, `resnet101`, `resnet152`
- DenseNet: `densenet121`, `densenet169`, `densenet201`, `densenet161`
- Inception: `inception_v3`
- MobileNet: `mobilenet_v2`

#### Sementic Segmentation
- UNet: `unet`: `unet34`, `unet101`, `unet152` (with resnet as backbone.)


### Dataloader
#### Classification
- [SAT-4 and SAT-6 airborne datasets](https:/csc.lsu.edu/~saikat/deepsat/)
- [EuroSat](http:/madm.dfki.de/downloads)
- [PatternNet](https:/sites.google.com/view/zhouwx/dataset)
- [NWPU_redisc45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html#)


## Showcase
If you extend this model to other datasets or build projects that use it, we'd love to hear from you.


## Reference
- [torchvision](https://github.com/pytorch/vision)

## Note
- If you are looking for the torchvision-enhance, please checkout the [enhance](https://github.com/sshuair/torchvision-enhance/tree/torchvision-enhance) branch. But it was deprecated.