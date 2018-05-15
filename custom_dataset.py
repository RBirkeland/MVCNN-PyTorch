import torch
from torch.utils.data.dataset import Dataset
import os
FOLDER_DATASET = "data/"
from PIL import Image
import numpy as np

class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        classes, class_to_idx = self.find_classes(root)

        self.classes = classes
        self.class_to_idx = class_to_idx

        self.transform = transform
        self.target_transform = target_transform

        # root / <label>  / <item> / <view>.png
        for label in os.listdir(root):
            for item in os.listdir(root + '/' + label):
                views = []
                for view in os.listdir(root + '/' + label + '/' + item):
                    views.append(root + '/' + label + '/' + item + '/' + view)

                self.x.append(views)
                self.y.append(class_to_idx[label])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        """
               Args:
                   index (int): Index
               Returns:
                   tuple: (sample, target) where target is class_index of the target class.
               """

        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view)
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)