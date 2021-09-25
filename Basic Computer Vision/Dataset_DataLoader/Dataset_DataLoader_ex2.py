from torchvision.datasets import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
import os

from tqdm import tqdm
import sys
from pathlib import Path
import requests

from skimage import io, transform
import matplotlib.pyplot as plt

import tarfile

class NotMNIST(VisionDataset):
    resource_url = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz'

    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool=False)-> None :
        super(NotMNIST, self).__init__(root, tansform=transform,
                                       target_transform=target_transform)
        if not self._check_exisits() or download:
            self.download()
        self.data, self.targets = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data[idx]
        image = io.imread(image_name)
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    # 파일명이 label(targets), 데이터 경로가 data에 담기는 구조
    def _load_data(self):
        filepath = self.image_folder
        data = []
        targets = []

        for target in os.listdir(filepath):
            # abspath : 절대경로 반환
            filenames = [os.path.abspath(
                os.path.join(filepath, target, x)
            ) for x in os.listdir(os.path.join(filepath, target))]

            targets.extend([target] * len(filenames))
            data.extend(filenames)
        return data, targets

    @property
    def raw_loader(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def image_folder(self) -> str:
        return os.path.join(self.root, 'notMNIST_large')

    def download(self) -> None:
        os.makedirs(self.raw_loader, exist_ok=True)
        fname = self.resource_url.split("/")[-1]
        chunk_size = 1024

        filesize = int(requests.head(self.resource_url).headers["Content-Length"])

        with requests.get(self.resource_url, stream=True) as r, open(
                os.path.join(self.raw_folder, fname), "wb") as f, tqdm(
            unit="B",  # unit string to be displayed.
            unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
            unit_divisor=1024,  # is used when unit_scale is true
            total=filesize,  # the total iteration.
            file=sys.stdout,  # default goes to stderr, this is the display on console.
            desc=fname  # prefix to be displayed on progress bar.
        ) as progress:
            for chunk in r.iter_content(chunk_size=chunk_size):
                # download the file chunk by chunk
                datasize = f.write(chunk)
                # on each chunk update the progress bar.
                progress.update(datasize)

        self._extract_file(os.path.join(self.raw_folder, fname), target_path=self.root)

    def _extract_file(self, fname, target_path) -> None:
        if fname.endswith("tar.gz"):
            tag = "r:gz"
        elif fname.endswith("tar"):
            tag = "r:"
        tar = tarfile.open(fname, tag)
        tar.extractall(path=target_path)
        tar.close()

    def _check_exists(self) -> bool:
        return os.path.exists(self.raw_folder)

dataset = NotMNIST("data", download=True)

# Datset 시각화

fig = plt.figure()

for i in range(8):
    sample = dataset[i]

    ax = plt.subplot(1, 4, i + 1) # 4장의 사진
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample[0])

    if i == 3:
        plt.show()
        break

# Transform
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

dataset = NotMNIST("data", download=False, transforms = data_transform)
dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=128, shuffle=True)
train_features, train_labels = next(iter(dataset_loader))
