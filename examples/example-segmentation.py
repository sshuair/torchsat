
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
from torchvision_x.transforms import transforms_seg,functional
from test import palette

def test_jpg():
    fig1 = plt.figure(figsize=(20,6))
    fig2 = plt.figure(figsize=(20,6))

    batch_size=4
    transform = transforms_seg.SegCompose([
    #     transforms_seg.SegFlip(),
        # transforms_seg.SegVFlip(), 
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
        
        # Variable, cuda
        # net feed forward
        # loss
        # back propagation
        
        # plot
        print(inputs.shape)
        for idx, item in enumerate(inputs):
            item = torch.squeeze(item, 0)
            img_ndarr = functional.to_ndarray(item, dtype='uint8')
            subplot = int(''.join(str(x) for x in [1, batch_size, idx + 1]))
            print(type(img_ndarr), img_ndarr.shape)
            img = Image.fromarray(img_ndarr)
            # target = Image.fromarray(img_ndarr)
            # tifffile.imshow(img_ndarr[:, :, [3, 2, 1]], figure=fig, subplot=subplot)
            tifffile.imshow(img_ndarr, figure=fig1, subplot=subplot) 
        plt.savefig('sample-data/transform_result/seg-jpg-input{}.png'.format(step), bbox_inches='tight')
        print('\n')
        for idx, item in enumerate(targets):
            item = torch.unsqueeze(item, 0)
            img_ndarr = functional.to_ndarray(item)
            subplot = int(''.join(str(x) for x in [1, batch_size, idx + 1]))
            print('target', type(item), item.shape)
            target_new = (item.numpy()).astype(np.uint8)
            target_new = np.squeeze(target_new,0)
            # img = Image.fromarray(target_new)
            # img.putpalette(palette.palette)  #设置颜色
            # img.save(new_tile_file)
            tifffile.imshow(target_new, figure=fig2, subplot=subplot)
            # tifffile.imshow(img_ndarr[:, :, [3, 2, 1]], figure=fig, subplot=subplot)
        fig2.savefig('sample-data/transform_result/seg-jpg-target{}.png'.format(step), bbox_inches='tight')
        print('\n')

def test_tif():
    fig1 = plt.figure(figsize=(20,6))
    fig2 = plt.figure(figsize=(20,6))

    batch_size=4
    transform = transforms_seg.SegCompose([
    #     transforms_seg.SegFlip(),
        # transforms_seg.SegVFlip(), 
        # transforms_seg.SegHFlip(),
        # transforms_seg.SegRandomFlip(),
        # transforms_seg.SegRandomRotate(90),
        # transforms_seg.SegRandomShift(40),
        # transforms_seg.SegRandomCrop((256,256)),
        # transforms_seg.SegCenterCrop(224),
        # transforms_seg.SegResize(224),
        # transforms_seg.SegPad(20),
        # transforms_seg.SegNoise(dtype='uint16', var=0.001),  #TODO
        # transforms_seg.SegGaussianBlur(sigma=2, dtype='uint16', multichannel=False),
        transforms_seg.SegPieceTransform(),
    #     transforms_seg.SegLambda(lambda x: functional.to_tensor(x))
        transforms_seg.SegToTensor(),
        # transforms_seg.SegNormalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

    trainset = image_loader.SemanticSegmentationLoader(
        rootdir='sample-data/', lstpath='sample-data/segmentation_tiff.lst',
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
        
        # plot
        print(inputs.shape)
        for idx, item in enumerate(inputs):
            item = torch.squeeze(item, 0)
            img_ndarr = functional.to_ndarray(item, dtype='uint16')
            subplot = int(''.join(str(x) for x in [1, batch_size, idx + 1]))
            print(type(img_ndarr), img_ndarr.shape)
            # img = Image.fromarray(img_ndarr)
            # target = Image.fromarray(img_ndarr)
            tifffile.imshow(img_ndarr[:, :, [4, 3, 2]], figure=fig1, subplot=subplot)
            # tifffile.imshow(img_ndarr, figure=fig1, subplot=subplot) 
        plt.savefig('sample-data/transform_result/seg-tif-input{}.png'.format(step), bbox_inches='tight')
        print('\n')
        for idx, item in enumerate(targets):
            item = torch.unsqueeze(item, 0)
            img_ndarr = functional.to_ndarray(item)
            subplot = int(''.join(str(x) for x in [1, batch_size, idx + 1]))
            print('target', type(item), item.shape)
            target_new = (item.numpy()).astype(np.uint8)
            target_new = np.squeeze(target_new,0)
            # img = Image.fromarray(target_new)
            # img.putpalette(palette.palette)  #设置颜色
            # img.save(new_tile_file)
            tifffile.imshow(target_new, figure=fig2, subplot=subplot)
            # tifffile.imshow(img_ndarr[:, :, [3, 2, 1]], figure=fig, subplot=subplot)
        fig2.savefig('sample-data/transform_result/seg-tif-target{}.png'.format(step), bbox_inches='tight')
        print('\n')


if __name__ == '__main__':
    # test_jpg()
    test_tif()