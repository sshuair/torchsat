# TorchSat (Work In Progress...)


TorchSat is an open-source PyTorch framework for satellite imagery analysis.

**Hightlight**:
- support multi-channels(> 3 channels, e.g. 8 channels) images and TIFF file as input.
- convenient data augmentation method for classification, sementic segmentation, object detection and instance segmentation.
- lots of common satellite datasets loader, .
- lots of models for satellite vision tasks, such as ResNet, PSPNet, SSD, and MaskRCNN ...
- training script for common satellite vision tasks.

# Install

python setup.py install


# How to use
- Introduction
- Data augmentation
- models
- train

# Features

## Transforms

We suppose all the input images and masks should be NumPy ndarray. The data shape should be **[height, width]** or **[height, width, channels]**.

### pixel level

Pixel-level transforms only change the input image and will leave any additional targets such as masks, bounding boxes unchanged. It support all channel images. Some transforms only support specific input channles.

| Transform            | Image  |  masks | BBoxes |
| -------------------- | :---:  |  :---: | :----: |
| ToTensor             |   ✓    |  /     |   /    |
| Normalize            |   ✓    |  /     |   /    |
| ToGray               |   ✓    |  /     |   /    |
| GaussianBlur         |   ✓    |  /     |   /    |
| RandomNoise          |   ✓    |  /     |   /    |
| RandomBrightness     |   TODO |  /     |   /    |
| RandomContrast       |   TODO |  /     |   /    |
| RandomGamma          |   TODO |  /     |   /    |
| RandomSigmoid        |   TODO |  /     |   /    |


### spatial-level
Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes. It support all channel images.

| Transform            | Image | masks | BBoxes |
| -------------------- | :---: | :---: | :----: |
| Resize               |   ✓   |   ✓   |        |
| Pad                  |   ✓   |   ✓   |        |
| RandomHorizontalFlip |   ✓   |   ✓   |        |
| RandomVerticalFlip   |   ✓   |   ✓   |        |
| RandomFlip           |   ✓   |   x   |        |
| CenterCrop           |   ✓   |   ✓   |        |
| RandomCrop           |   ✓   |   ✓   |        |
| RandomResizedCrop    |   ✓   |   ✓   |        |
| ElasticTransform     |   ✓   |   ✓   |        |
| RandomRotation       |   ✓   |   ✓   |        |
| RandomShift          |   ✓   |   ✓   |        |



## Dataloader
### Classification
- Sat
- patternnet
- NWPU_redisc45
- EuroSat

### Sementic Segmentation


### Object Detection

### Instance Segmentation

## Models
### Classification

### Sementic Segmentation


### Object Detection

### Instance Segmentation

