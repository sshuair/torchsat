import os

from .folder import DatasetFolder
from .utils import pil_loader, tifffile_loader

CLASSES_TO_IDX = {
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


class EuroSAT(DatasetFolder):
    url_rgb = 'http://madm.dfki.de/files/sentinel/EuroSAT.zip'
    url_allband = 'http://madm.dfki.de/files/sentinel/EuroSATallBands.zip'

    def __init__(self, root, mode='RGB', download=False, **kwargs):
        if mode not in ['RGB', 'AllBand']:
            raise ValueError('{} is not suppport mode, replace with RGB or AllBand'.format(self.mode))

        if mode == 'RGB':
            self.loader = pil_loader
            self.extensions = ['.jpg', '.jpeg']
        else:
            self.loader = tifffile_loader
            self.extensions = ['.tif', '.tiff']
            root = os.path.join(root, 'images/remote_sensing/otherDatasets/sentinel_2/tif')

        classes = list(CLASSES_TO_IDX.keys())

        super(EuroSAT, self).__init__(root, self.loader, self.extensions, 
        classes=classes, class_to_idx=CLASSES_TO_IDX, **kwargs
        )

    def download():
        pass
