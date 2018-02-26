
import sys
sys.path.append('/Users/sshuair/deep-learning/vision-multi')
from skimage.external import tifffile
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, filters, util
from torchvision_x.datasets import image_loader
from torchvision_x.transforms import transforms_cls,functional

def test_jpg():
    fig = plt.figure(figsize=(20,6))

    batch_size=4
    transform = transforms_cls.Compose([
    #     transforms_cls.Flip(),
        transforms_cls.VFlip(), 
        transforms_cls.HFlip(),
        transforms_cls.RandomFlip(),
        transforms_cls.RandomRotate(90),
        transforms_cls.RandomShift(40),
        transforms_cls.RandomCrop((256,256)),
        transforms_cls.CenterCrop(224),
        transforms_cls.Resize(224),
        transforms_cls.Pad(20),
        # transforms_cls.Noise(dtype='uint16', var=0.001),  #TODO
        transforms_cls.GaussianBlur(sigma=2, dtype='uint8', multichannel=False),
        transforms_cls.PieceTransform(),
    #     transforms_cls.Lambda(lambda x: functional.to_tensor(x))
        transforms_cls.ToTensor(),
        # transforms_cls.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

    trainset = image_loader.SingleLabelImageLoader(
        rootdir='sample-data/', lstpath='sample-data/classification_jpg.lst',
        filetype='jpg', transform=transform,
        )
    trainloader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=False)

    for step, (inputs, targets) in enumerate(trainloader):
        print('batch: {} ........'.format(step))
        print(type(inputs), inputs.shape)
        print(type(targets), targets.shape)
        
        # Variable, cuda
        # net feed forward
        # loss
        # back propagation
        
        #plot
        for idx, item in enumerate(inputs):
            item = torch.squeeze(item,0)
            img_ndarr = functional.to_ndarray(item, dtype='uint8')
            subplot = int(''.join(str(x) for x in [1, batch_size, idx+1]))
            print(type(img_ndarr), img_ndarr.size)
            # tifffile.imshow(img_ndarr[:,:,[3,2,1]], figure=fig, subplot=subplot)  
            tifffile.imshow(img_ndarr, figure=fig, subplot=subplot)  
        plt.savefig('sample-data/transform_result/cls_jpg_{}.png'.format(step), bbox_inches='tight')
        print('\n')

def test_tif():
    fig = plt.figure(figsize=(20,6))

    batch_size=4
    transform = transforms_cls.Compose([
    #     transforms_cls.Flip(),
        transforms_cls.VFlip(), 
        transforms_cls.HFlip(),
        transforms_cls.RandomFlip(),
        transforms_cls.RandomRotate(90),
        transforms_cls.RandomShift(40),
        transforms_cls.RandomCrop((256,256)),
        transforms_cls.CenterCrop(224),
        transforms_cls.Resize(224),
        transforms_cls.Pad(20),
        # transforms_cls.Noise(dtype='uint16', var=0.001),  #TODO
        transforms_cls.GaussianBlur(sigma=2, dtype='uint16', multichannel=False),
        transforms_cls.PieceTransform(),
    #     transforms_cls.Lambda(lambda x: functional.to_tensor(x))
        transforms_cls.ToTensor(),
        # transforms_cls.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

    trainset = image_loader.SingleLabelImageLoader(
        rootdir='sample-data/', lstpath='sample-data/classification_tiff.lst',
        filetype='tif', transform=transform,
        )
    trainloader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=False)

    for step, (inputs, targets) in enumerate(trainloader):
        print('batch: {} ........'.format(step))
        print(type(inputs), inputs.shape)
        print(type(targets), targets.shape)
        
        # Variable, cuda
        # net feed forward
        # loss
        # back propagation
        
        #plot
        for idx, item in enumerate(inputs):
            item = torch.squeeze(item,0)
            img_ndarr = functional.to_ndarray(item, dtype='uint16')
            subplot = int(''.join(str(x) for x in [1, batch_size, idx+1]))
            print(type(img_ndarr), img_ndarr.size)
            tifffile.imshow(img_ndarr[:,:,[3,2,1]], figure=fig, subplot=subplot)  
            # tifffile.imshow(img_ndarr, figure=fig, subplot=subplot)  
        plt.savefig('sample-data/transform_result/cls_tif_{}.png'.format(step), bbox_inches='tight')
        print('\n')

if __name__ == '__main__':
    test_jpg()
    test_tif()