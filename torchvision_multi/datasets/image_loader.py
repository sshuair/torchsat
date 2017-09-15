import csv
import os

import torch
import numpy as np
from PIL import Image
from skimage.external import tifffile
from torch.utils.data import DataLoader, Dataset

IMG_EXTENSIONS = [
    'jpg', 'JPG', 'jpeg', 'JPEG',
    'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP',
]

TIF_EXTENSIONS = [
    'tif', 'tiff', 'TIF', 'TIFF'
]


def _image_loader(path, filetype):
    """
    loading image from path
    :param path: image path
    :param filetype: image type
    :return: ndarray image
    """
    if filetype in IMG_EXTENSIONS:
        img = np.array(Image.open(path))
        return img
    elif filetype in TIF_EXTENSIONS:
        img = tifffile.imread(path)
        img = img.astype(dtype=np.int32) # torch.from_numpy only supported types are: double, float, int64, int32, and uint8.
        return img


def parse_single_lstfile(rootdir, lstpath):
    """
    parse image file into tuple for single-label classification
    file organization：
        class \t file_path
        4  \t plane.jpg
        1  \t human.jpg
        ... \t ...
    :param rootdir: train/val 文件路径
    :param lstpath: 类别以及文件对应关系文件
    :return: images(image_path, class)
    """
    images_path = []
    with open(lstpath,'r',) as f:
        for line in csv.reader(f, delimiter="\t"):
            _class = int(line[0])
            filename = line[1]
            if os.path.exists(os.path.join(rootdir, filename)):
                item = (os.path.join(rootdir, filename), _class)
                images_path.append(item)
    return images_path


def parse_multi_lstfile(rootdir, lstpath):
    """
    parse image file into tuple for multi-label classification
    file organization：
        class1 \t class2 ... file_path
        1 \t 0 \t1 ... plane_people.jpg

    :param rootdir: train/val 文件路径
    :param lstpath: 类别以及文件对应关系文件
    :return: images(image_path, class)
    """
    images_path = []
    with open(lstpath, 'r',) as f:
        for line in csv.reader(f, delimiter="\t"):
            _class = np.array([int(x) for x in line[0:-1]]) # 必须是np.array, 不能用list，否则enumerate时target不是tensor
            filename = line[-1]
            if os.path.exists(os.path.join(rootdir, filename)):
                item = (os.path.join(rootdir, filename), _class)
                images_path.append(item)
    return images_path


def parse_segmentation_lstfile(rootdir, lstpath):
    """
    read train.txt file and parse into image list and label list.
    :return: images path list and labels path list
    """
    images_path = []
    with open(lstpath, 'r', ) as f:
        for line in csv.reader(f, delimiter="\t"):
            image = os.path.join(rootdir, line[0])
            label = os.path.join(rootdir, line[1])
            if os.path.exists(image) and os.path.exists(label):
                item = (image, label)
                images_path.append(item)
    return images_path



class SingleLabelImageLoader(Dataset):
    """
    single-label classification: single label, file list
    """
    def __init__(self, root, lstpath, filetype='jpg', transform=None, target_transform=None, loader=_image_loader):
        images_path = parse_single_lstfile(root, lstpath)
        if len(images_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.images_path = images_path
        self.filetype = filetype
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        image, target = self.images_path[index]
        image = self.loader(image, self.filetype)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images_path)


class MultiLabelImageLoader(Dataset):
    """
    multi-label classification: multi label, file list
    """
    def __init__(self, root, lstpath, filetype='jpg', transform=None, target_transform=None, loader=_image_loader):
        images_path = parse_multi_lstfile(root, lstpath)
        if len(images_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.images_path = images_path
        self.filetype = filetype
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        image, target = self.images_path[index]
        image = self.loader(image, self.filetype)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images_path)


class ImageFolderDet(Dataset):
    """
    Object Detection, 对象检测
    """
    pass


class SemanticSegmentationLoader(Dataset):
    """
    Semantic Segmentation Loader 
    """

    def __init__(self, rootdir, lstpath, filetype='jpg', transform=None, target_transform=None, loader=_image_loader):
        images_path = parse_segmentation_lstfile(rootdir, lstpath)
        if len(images_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.images_path = images_path
        self.rootdir = rootdir
        self.lstpath = lstpath
        self.filetype = filetype
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # self.images, self.labels = self.parse_file_path()
        pass


    # def __getitem__(self, index):
    #     image_path = self.images[index]
    #     label_path = self.labels[index]
    #     image = self.loader(image_path, self.filetype)
    #     label = self.loader(label_path, 'png')
    #     label = label[:, :, np.newaxis]  #特殊处理 for transform_enhance
    #
    #     if self.transform is not None:
    #         image, label = self.transform([image, label])
    #         label = label.view(label.shape[1], label.shape[2]).long()  # 经过transform_enhance变换后，处理成width、height，LongTensor
    #     if self.target_transform is not None:
    #         label = self.target_transform(label)
    #
    #     return image, label

    def __getitem__(self, index):
        # image_path = self.images[index]
        # label_path = self.labels[index]
        image, target = self.images_path[index]
        image = self.loader(image, self.filetype)
        target = self.loader(target, 'png')

        if self.transform is not None:
            image, target = self.transform(image, target)
            target = torch.from_numpy(target).long()
            # label = label.view(label.shape[1], label.shape[2]).long()  # 经过transform_enhance变换后，处理成width、height，LongTensor
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        return image, target

    def __len__(self):
        return len(self.images_path)


    def parse_file_path(self):
        """
        read train.txt file and parse into image list and label list.
        :return: images path list and labels path list
        """
        images = []
        labels = []
        with open(os.path.join(self.rootdir, self.filelst), 'r', ) as f:
            for line in csv.reader(f, delimiter=","):
                image = os.path.join(self.rootdir, line[0])
                label = os.path.join(self.rootdir, line[1])
                if os.path.exists(image) and os.path.exists(label):
                    images.append(image)
                    labels.append(label)
        return images,labels




if __name__ == '__main__':
    
    root = '/opt/project/projects/landsat8_river'
    filelst = 'train.txt'
    ss = SemanticSegmentation(rootdir=root, filelst = filelst)

    input_channel = 3
    target_channel = 3

    transform = transforms_enhance.EnhancedCompose([
        transforms_enhance.Merge(),
        transforms_enhance.RandomCropNumpy(size=(256, 256)),
        transforms_enhance.RandomRotate(),
        transforms_enhance.ElasticTransform(alpha=1000, sigma=30),
        transforms_enhance.Split([0, input_channel], [input_channel, input_channel + target_channel]),
        [transforms_enhance.CenterCropNumpy(size=(224, 224)), transforms_enhance.CenterCropNumpy(size=(224, 224))],
        # [NormalizeNumpy(), MaxScaleNumpy(0, 1.0)],
        # for non-pytorch usage, remove to_tensor conversion
        [transforms.Lambda(transforms_enhance.to_tensor), transforms.Lambda(transforms_enhance.to_tensor)]
    ])

    trainloader = DataLoader(
        dataset=SemanticSegmentation(
            rootdir='/opt/project/projects/landsat8_river', filelst='train.txt', filetype='tif',
            transform=transform,
        ),
        batch_size=16,
        shuffle=True,
    )

    for idx, (data, target) in enumerate(trainloader):
        print(data.shape)
        print(target.shape)
        # if idx == 0:
        #     img = torchvision.utils.make_grid(imgs).numpy()
        #     img = np.transpose(img, (1, 2, 0))
        #     img = img[:, :, ::-1]
        #     plt.imshow(img)
        #     plt.show()
        #     plt.imshow(dst.decode_segmap(labels.numpy()[i+1]))
        #     plt.show()
        pass
    pass