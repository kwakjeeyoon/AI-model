# Dataset & DataLoader
### 기본 구조
```python
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        sample = {"Data":data, "Class":label}
        return sample
MyDataset = CustomDataset(data, labels) # Dataset
```
### Transform
```python
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

MyDataset = CustomDataset(data, labels, transforms = data_transform) # Dataset + transform
MyDataLoader = DataLoader(MyDataset, batch_size=128, shuffle=True)

```
### Dataset 시각화
```python
# 한장씩 보는 경우
train_features, train_labels = next(iter(MyDataLoader))

# 여러 개의 data를 보는 경우
fig = plt.figure()

# 4장의 이미지 보여줌
for i in range(8):
    sample = MyDataset[i]

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample[0])

    if i == 3:
        plt.show()
        break
```