import os
from .utils import image_loader

class SegDataset(object):
    """SegDataset  This class is for common semantic segmentation scenario. 
    Data organization adopted from PASCAL VOC datasets. http://host.robots.ox.ac.uk/pascal/VOC/index.html

    The trainval.txt, train.txt or val.txt should organize by:
    :: 
      
        relative_path/train_image/a0001.jpg\trelative_path/mask_image/a0001.png
        relative_path/train_image/a0002.jpg\trelative_path/mask_image/a0002.png
        relative_path/train_image/a0003.jpg\trelative_path/mask_image/a0003.png

    the mask image should be gray sacle, value from 0~255, each value represent one class.

    Args:
        root (str): root dir of the datasets
    
    """
    def __init__(self, root, image_set='train.txt', transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform

        with open(os.path.join(root, image_set)) as f:
            image_files = [x for x in f.readlines()]

        self.images = [os.path.join(root, x.split('\t')[0].strip()) for x in image_files]
        self.masks = [os.path.join(root, x.split('\t')[1].strip()) for x in image_files]

    def __getitem__(self, index):
        img = image_loader(self.images[index])
        target = image_loader(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.images)
