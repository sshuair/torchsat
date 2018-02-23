import csv
import os

import torch
import numpy as np
from PIL import Image
from skimage.external import tifffile
from torch.utils.data import DataLoader, Dataset
import warnings

IMG_EXTENSIONS = [
    'jpg', 'JPG', 'jpeg', 'JPEG',
    'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP',
]

TIF_EXTENSIONS = [
    'tif', 'tiff', 'TIF', 'TIFF'
]


def _image_loader(path, filetype):
    """
    loading image from the given path

    :param path: image path
    :param filetype: image type
    :return: ndarray image
    """
    if filetype in IMG_EXTENSIONS:
        img = np.array(Image.open(path))
        return img
    elif filetype in TIF_EXTENSIONS:
        img = tifffile.imread(path)
        # warnings.warn("tiff file should be 16-bit format.")
        img = img.astype(dtype=np.int32) # torch.from_numpy only supported types are: double, float, int64, int32, and uint8.
        return img


def parse_single_lstfile(rootdir, lstpath, sep='\t'):
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
        for line in csv.reader(f, delimiter=sep):
            _class = int(line[0])
            filename = line[1]
            if os.path.exists(os.path.join(rootdir, filename)):
                item = (os.path.join(rootdir, filename), _class)
                images_path.append(item)
    return images_path


def parse_multi_lstfile(rootdir, lstpath, sep='\t'):
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
        for line in csv.reader(f, delimiter=sep):
            class_ = np.array([int(x) for x in line[0:-1]]) #必须是np.array, 不能用list，否则enumerate时target不是tensor
            filename = line[-1]
            if os.path.exists(os.path.join(rootdir, filename)):
                item = (os.path.join(rootdir, filename), class_)
                images_path.append(item)
    return images_path


def parse_segmentation_lstfile(rootdir, lstpath, sep='\t'):
    """
    read train.txt file and parse into image list and label list.
    :return: images path list and labels path list
    """
    images_path = []
    with open(os.path.join(rootdir, lstpath), 'r', ) as f:
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
    def __init__(self, rootdir, lstpath, filetype='jpg', sep='\t', transform=None, target_transform=None, loader=_image_loader):
        images_path = parse_single_lstfile(rootdir, lstpath, sep=sep)
        if len(images_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + rootdir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.rootdir = rootdir
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
    def __init__(self, rootdir, lstpath, filetype='jpg', sep='\t', transform=None, target_transform=None, loader=_image_loader):
        images_path = parse_multi_lstfile(rootdir, lstpath,sep=sep)
        if len(images_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + rootdir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.rootdir = rootdir
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
    Object Detection Loader
    """
    pass


class SemanticSegmentationLoader(Dataset):
    """
    Semantic Segmentation Loader.
    the input transform will work for input image and target image both
    """

    def __init__(self, rootdir, lstpath, filetype=['jpg', 'png'], sep='\t', transform=None, target_transform=None, loader=_image_loader):
        images_path = parse_segmentation_lstfile(rootdir, lstpath, sep=sep)
        if len(images_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + rootdir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.images_path = images_path
        self.rootdir = rootdir
        self.lstpath = lstpath
        self.filetype = filetype
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        image, target = self.images_path[index]
        image = self.loader(image, self.filetype[0])
        target = self.loader(target, self.filetype[1])

        if self.transform is not None:
            image, target = self.transform(image, target)
            # target = torch.from_numpy(target).long()
        # if self.target_transform is not None:
        #     label = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.images_path)
