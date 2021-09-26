from sklearn.datasets import load_iris

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

class IrisDataset(Dataset):
    def __init__(self):
        iris = load_iris()
        self.X = iris['data']
        self.y = iris['target']

        self.feature_names = iris['feature_names']
        self.target_names = iris['target_names']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype = torch.float)
        y = torch.tensor(self.y[idx], dtype = torch.long)
        return X,y

dataset_iris = IrisDataset()
# print(len(dataset_iris))
# print(dataset_iris[0])

# batch_size, shuffle, sampler/batch_sampler
dataloader_iris = DataLoader(dataset_iris,
                                     batch_size=4,
                                     shuffle=True)
next(iter(dataloader_iris))