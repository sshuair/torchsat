# TorchSat (Work In Progress...)

TorchSat is an open-source PyTorch framework for satellite imagery analysis.

**Hightlight**:
- Support multi-channels(> 3 channels, e.g. 8 channels) images and TIFF file as input.
- Convenient data augmentation method for classification, sementic segmentation and object detection.
- Lots of common satellite datasets loader, .
- Lots of models for satellite vision tasks, such as ResNet, UNet, PSPNet, SSD, FasterRCNN ...
- Training script for common satellite vision tasks.

# Install

python setup.py install

# How to use
- [Introduction]()
- [Data Augmentation]()
- models
- train

# Features

## Transforms

We suppose all the input images and masks should be NumPy ndarray. The data shape should be **[height, width]** or **[height, width, channels]**.

### pixel level

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

### spatial-level
Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes. It support all channel images.

| Transform            | Image | masks | BBoxes |
| -------------------- | :---: | :---: | :----: |
| Resize               |   ✓   |   ✓   |        |
| Pad                  |   ✓   |   ✓   |        |
| RandomHorizontalFlip |   ✓   |   ✓   |    ✓   |
| RandomVerticalFlip   |   ✓   |   ✓   |    ✓   |
| RandomFlip           |   ✓   |   ✓   |    ✓   |
| CenterCrop           |   ✓   |   ✓   |        |
| RandomCrop           |   ✓   |   ✓   |        |
| RandomResizedCrop    |   ✓   |   ✓   |        |
| ElasticTransform     |   ✓   |   ✓   |        |
| RandomRotation       |   ✓   |   ✓   |        |
| RandomShift          |   ✓   |   ✓   |        |



## Models
### Classification
All models support multi-channels as input (e.g. 8 channels).
- AlexNet
- VGG
- ResNet
- DenseNet
- Inception

### Sementic Segmentation
- UNet(TODO)
- PSPNet(TODO)

### Object Detection

- SSD(TODO)
- YOLOV3(TODO)
- FasterRCNN(TODO)

## Dataloader
### Classification
- [SAT-4 and SAT-6 airborne datasets](https:/csc.lsu.edu/~saikat/deepsat/)
- [EuroSat](http:/madm.dfki.de/downloads)
- [PatternNet](https:/sites.google.com/view/zhouwx/dataset)
- NWPU_redisc45


### Sementic Segmentation


### Object Detection
