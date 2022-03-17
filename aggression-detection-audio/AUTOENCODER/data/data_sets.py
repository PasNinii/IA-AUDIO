"""
Dataset in charge of knowing where the source of data, labels and transforms.
Should provide access to the data by indexing.
"""

import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch.utils.data as data

class Dataset(data.Dataset):
    """
        * Implement pytorch data.Dataset
    """
    def __init__(self, data, loading_function, transforms=None, CNN=False):
        self.data = data
        self.transforms = transforms
        self.loading_function = loading_function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        X, y, c = self.loading_function(data['path']), data['label'], data['classID']
        if self.transforms is not None:
            sound, sr, label = self.transforms.apply(X, y)
            return sound, sr, y, c
        return X, y, c









