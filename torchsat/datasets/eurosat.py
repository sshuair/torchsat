import os
from collections import namedtuple
from PIL import Image
import torch.utils.data as data
from torchvision.datasets.folder import DatasetFolder

from .utils import download_url


LANDUSE_CLASSES = {
        'AnnualCrop': 0,
        'Forest': 1,
        'HerbaceousVegetation': 2,
        'Highway': 3,
        'Industrial': 4,
        'Pasture': 5,
        'PermanentCrop': 6,
        'Residential': 7,
        'River': 8,
        'SeaLake': 9,
    }

class EuroSAT(data.Dataset):

    def __init__(self, root, mode='RGB', transform=None, target_transform=None, 
        download=False):
        self.root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        if mode not in ['RGB', 'AllBand']:
            raise ValueError('{} is not suppport mode, replace with RGB or AllBand'.format(self.mode))

        if download:
            self.download()

        self.images = []
        self.targets = []

        for landuse in os.listdir(self.root):
            for img in os.listdir(os.path.join(self.root, landuse)):
                self.targets.append(LANDUSE_CLASSES[landuse])
                self.images.append(os.path.join(self.root, landuse, img))

    def __getitem__(self, index):

        if self.mode == 'RGB':
            image = pil_loader(self.images[index])
            target = self.targets[index]
        elif self.mode == 'AllBand':
            raise ValueError('not implemented')
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target

    def __len__(self):
        return len(self.images)
    
    def _check_integrity(self):
        pass

    def download():
        pass

    def __repr__(self):
        pass


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def tiffile_loader(path):
    import tiffifle
    return tifffile.imread(path)