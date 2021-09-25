import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.data = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.data[idx]
        sample = {"Text":text, "Class":label}
        return sample

text = ['Happy', 'Amazing', 'Sad', 'Unhapy', 'Glum']
labels = ['Positive', 'Positive', 'Negative', 'Negative', 'Negative']
MyDataset = CustomDataset(text, labels)

# print(type(MyDataset)) # <class '__main__.CustomDataset'>

MyDataLoader = DataLoader(MyDataset, batch_size=2, shuffle=True)
next(iter(MyDataLoader))

for dataset in MyDataLoader:
    print(dataset)
# {'Text': ['Amazing', 'Unhapy'], 'Class': ['Positive', 'Negative']}
# {'Text': ['Happy', 'Glum'], 'Class': ['Positive', 'Negative']}
# {'Text': ['Sad'], 'Class': ['Negative']}

