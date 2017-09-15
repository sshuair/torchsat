import torch
from torch.utils.data import Dataset, DataLoader
from skimage.external import tifffile
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
from torchvision_multi import transform_multi
from torchvision_multi.datasets import image_loader


# fig = plt.figure(figsize=(20, 6))
# img_tif = tifffile.imread('./sample-data/7-channel.tif')
# img_tif_addnoise = transform_multi.noise(img_tif,16,0,0.001)
#
# img_jpg = Image.open('./sample-data/2007_000129.jpg')
# img_tif_addnoise = transform_multi.noise(img_tif,16)


batch_size = 2
transform = transform_multi.SegCompose([
    transform_multi.SegRandomRotate(0.4)
    # transform_multi.RandomShift(0.5, 30, 30),
    # transform_multi.RandomCrop((200,200)),
    # transform_multi.SegRandomNoise(0.5,16),

    # transform_multi.Lambda(lambda x: transform_multi.to_tensor(x))
])

trainset = image_loader.SemanticSegmentationLoader(
    rootdir='./sample-data/', lstpath='./sample-data/segmentation.lst',
    filetype='jpg', transform=transform,
)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)

for step, (inputs, targets) in enumerate(trainloader):
    #     print('batch: {} ........'.format(idx))
    #     print(type(images), images.shape)
    #     print(type(targets), targets.shape)

    # Variable, cuda
    # net feed forward
    # loss
    # back propagation

    # plot
    for idx, item in enumerate(inputs):
        item = torch.squeeze(item, 0)
        img_ndarr = transform_multi.to_ndarray(item)
        # subplot = int(''.join(str(x) for x in [1, batch_size, idx + 1]))
        print(type(img_ndarr), img_ndarr.size)
        # tifffile.imshow(img_ndarr[:, :, [3, 2, 1]], figure=fig, subplot=subplot)
    # plt.savefig('./sample-data/plot/{}.png'.format(step), bbox_inches='tight')
    print('\n')