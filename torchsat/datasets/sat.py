import os

import numpy as np
import torch.utils.data as data

from scipy.io import loadmat


class SAT(data.Dataset):
    """SAT-4 and SAT-6 datasets
    
    Arguments:
        data {root} -- [description]
    
    Raises:
        ValueError -- [description]
        ValueError -- [description]
    
    Returns:
        [type] -- [description]
    """

    classes_sat4 = {"barren land": 0, "trees": 1, "grassland": 2, "none": 3}
    classes_sat6 = {
        "building": 0,
        "barren land": 1,
        "trees": 2,
        "grassland": 3,
        "road": 4,
        "water": 5,
    }

    def __init__(
        self,
        root,
        mode="SAT-4",
        image_set="train",
        download=False,
        transform=False,
        target_transform=False,
    ):
        if mode not in ("SAT-4", "SAT-6"):
            raise ValueError("Argument mode should be 'SAT-4' or 'SAT-6'")

        if image_set not in ("train", "test"):
            raise ValueError("Argument image_set should be 'train' or 'test'")

        if download:
            download()

        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        _mat_path = os.path.join(self.root, self.mode.lower() + "-full.mat")
        self._mat = loadmat(_mat_path)

        self.images, self.targets = self._load_data(self._mat, self.image_set)

    def __getitem__(self, index):
        image = self.images[:, :, :, index]
        target = np.argmax(self.targets[:, index], axis=0)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)

    def _load_data(self, mat, image_set):
        if image_set == "train":
            images = mat["train_x"]
            targets = mat["train_y"]
        else:
            images = mat["test_x"]
            targets = mat["test_y"]
        return images, targets

    def download(self):
        pass
