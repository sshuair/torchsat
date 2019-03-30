import os
from .folder import DatasetFolder, tifffile_loader, pil_loader


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

    def __init__(self, root, mode='RGB', download=False, **kwargs):
        if mode not in ['RGB', 'AllBand']:
            raise ValueError('{} is not suppport mode, replace with RGB or AllBand'.format(self.mode))

        if mode == 'RGB':
            self.loader = pil_loader
            self.extensions = ['.jpg', '.jpeg']
        else:
            self.loader = tifffile_loader
            self.extensions = ['.tif', '.tiff']

        classes = list(CLASSES_TO_IDX.keys())

        super(EuroSAT, self).__init__( root, self.loader, self.extensions, 
        classes=classes, class_to_idx=CLASSES_TO_IDX, **kwargs
        )

    def download():
        pass


if __name__ == "__main__":
    root_fp = '/Volumes/sshuair/dl-satellite-data/pytorch-satallite-data/classification/EuroSAT'
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    eurosat_dataset = eurosat.EuroSAT(root_fp, mode='RGB', transform=transform)
    eurosat_loader = DataLoader(eurosat_dataset)
    for inputs, labels in eurosat_loader:
        print(inputs, labels)
        break