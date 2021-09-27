from torch.utils.data import Dataset
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, df, transform, target):
        self.df = df
        self.transform = transform
        self.target = target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df['path'].iloc[idx]
        label = self.df[self.target].iloc[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label

class EvalDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df['path'].iloc[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        return image

