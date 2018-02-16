# torchvision-enhance

torchvision-enhance is used to enhance the offical PyTorch vision library torchvision. Here is the enhanced parts:
- support multi-channel(> 4 channels, e.g. 8 channels) images
- support 16-bit TIF file 
- more easier to semantic segmentation transform



## support transforms
- RandomFlip
- RandomVFlip
- RandomHFlip
- RandomRotate
- RandomShift
- RandomCrop
- CenterCrop
- Resize
- Pad
- GaussianBlur
- PieceTransform
- Lambda
- ToTensor
- Normalize

## install
```
pip install torchvision-enhance
```  

or  install from the source

```
git clone 
pip install -r requirements.txt
python setup.py install
```

## usage
For more useage, check out the [example-classification.py](./test/example-classification.py) and [example-segmentation.py](./test/example-segmentation.py)

``` python
from torchvision_x.datasets import image_loader
from torchvision_x.transforms import transforms_seg,functional

transform = transforms_seg.SegCompose([
#     transforms_seg.SegFlip(),
    transforms_seg.SegVFlip(), 
    # transforms_seg.SegHFlip(),
    # transforms_seg.SegRandomFlip(),
    # transforms_seg.SegRandomRotate(90),
    # transforms_seg.SegRandomShift(40),
    # transforms_seg.SegRandomCrop((256,256)),
    # transforms_seg.SegCenterCrop(224),
    # transforms_seg.SegResize(224),
    # transforms_seg.SegPad(20),
    # transforms_seg.SegNoise(dtype='uint16', var=0.001),  #TODO
    # transforms_seg.SegGaussianBlur(sigma=2, dtype='uint8', multichannel=False),
    # transforms_seg.SegPieceTransform(),
#     transforms_seg.SegLambda(lambda x: functional.to_tensor(x))
    transforms_seg.SegToTensor(),
    transforms_seg.SegNormalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

trainset = image_loader.SemanticSegmentationLoader(
    rootdir='sample-data/', lstpath='sample-data/segmentation_jpg.lst',
    filetype='jpg', transform=transform,
    )
trainloader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=False)

for step, (inputs, targets) in enumerate(trainloader):
    print('batch: {} ........'.format(step))
    print(type(inputs), inputs.shape)
    print(type(targets), targets.shape)
```

## TODO
- Noise
