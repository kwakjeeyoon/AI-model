# train / val - trasnform 다르게 만들기
# 최고 accuracy, loss 만 pt로 저장하기
# early stopping 넣기
# + 외부 데이터셋 사용하기
# -------------------------------------------------
# <완료>
#
# train / test 분리-> validation score 계산하기
# normalize -> 평균 표준편차 구하기
# save_model 코드 넣기
# accuracy 넣기
# F1 Score 넣기

import pickle
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from time import time

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

# from albumentations import *
# from albumentations.pytorch import ToTensorV2

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Data preprocessing
train_base_path = '/opt/ml/input/data/train/images/'
test_base_path = '/opt/ml/input/data/eval/'

# train/test 데이터 개수
mask_file = ['mask1.jpg','mask2.jpg','mask3.jpg','mask4.jpg','mask5.jpg','incorrect_mask.jpg', 'normal.jpg']
train_label = []
train_input_dir = []
for folder in os.listdir(train_base_path):
    if not folder.startswith('.'):
        id, gender, race, age = folder.split('_')
        label = 0
        for i, file in enumerate(mask_file):
            if not file.startswith('.'):
                train_input_dir.append(train_base_path + '/' +folder+'/'+file)
                if gender == 'female':
                    label += 3
                if 30 <= int(age) < 60:
                    label +=1
                elif int(age) >= 60:
                    label += 2
                if i == 5:
                    label += 6
                elif i == 6:
                    label += 12
                train_label.append(label)
test_input_dir = test_base_path + 'images'
print(len(train_input_dir))
print(len(os.listdir(test_input_dir)))

# 원본 데이터 크기
example = np.array(Image.open(list(train_input_dir)[0]))
example = torch.tensor(example)
example.shape
torch.tensor(example).shape

mean, std = (0.56019358, 0.52410121, 0.501457), (0.23318603, 0.24300033, 0.24567522)

# def get_ext(img_dir, img_id):
#     filename = os.listdir(os.path.join(img_dir, img_id))[0]
#     ext = os.path.splitext(filename)[-1].lower()
#     return ext
def get_img_stats(img_ids):
    img_info = dict(heights=[], widths=[], means=[], stds=[])
    for img_id in tqdm(img_ids):
        img = np.array(Image.open(img_id))
        h, w, _ = img.shape
        img_info['heights'].append(h)
        img_info['widths'].append(w)
        img_info['means'].append(img.mean(axis=(0,1)))
        img_info['stds'].append(img.std(axis=(0,1)))
    return img_info

# img_info = get_img_stats(train_input_dir)

# print(f'RGB Mean: {np.mean(img_info["means"], axis=0) / 255.}')
# print(f'RGB Standard Deviation: {np.mean(img_info["stds"], axis=0) / 255.}')

# Dataset
class MyDataset(Dataset):
    def __init__(self, img_path, labels, transform=None, train_transform=None, val_transform=None):
        self.transform = transform
        self.labels = labels
        self.img_paths = img_path

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        y = self.labels[index]
        X = Image.open(self.img_paths[index])

        if self.transform:
            X = self.transform(X)

        return X, y

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    Normalize(mean=(0.56019358, 0.52410121, 0.501457), std=(0.23318603, 0.24300033, 0.24567522)),
#     ToTensor(),
#     albu.Resize(64, 64)
#     transforms.functional.posterize(), # 효과를 잘 모르겠네..? -> 이미지 채널 당 사용되는 음영의 개수를 제한 (이미지 색상이 단순화됨)
])

# Hyper parameter
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH ="saved"
num_workers = 4
num_classes = 18

dataset = MyDataset(img_path=list(train_input_dir),labels=train_label,transform=transform)

n_val = int(len(dataset) * 0.2)
n_train = len(dataset) - n_val
train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = num_workers)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = num_workers)

X, y = next(iter(train_loader))

from torchvision import models

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(MyModel, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1000, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.vgg19(x)
#         x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

from torchvision import models
from efficientnet_pytorch import EfficientNet

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.EfficientNet = EfficientNet.from_pretrained('efficientnet-b2', num_classes=18)

    def forward(self, x):
        x = self.EfficientNet(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = MyModel(num_classes)
model = model.to(device)

print(model)

# Model
def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


def accuracy(y_pred, y_test):
    _, predicted = torch.max(y_pred.data, 1)

    total += y_test.size(0)
    correct += (predicted == y_test).sum().item()
    return correct


model = MyModel(num_classes).to(device)
# 학습 재시작
# checkpoint = torch.load("saved/0826_56_0.52019_0.15830_51.59494_0.800.pt")
# model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# for param in model.parameters():
#     param.requires_grad = False
# for param in model.classifier.parameters():
#     param.requires_grad = True

for n, p in model.named_parameters():
    if '_fc' not in n:
        p.requires_grad = False

model = torch.nn.DataParallel(model)

# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

for e in range(1, EPOCHS + 1):
    #     model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    val_loss = 0
    for X_batch, y_batch in tqdm(train_loader, leave=True):
        #         time.sleep(0.25)

        #         X_batch, y_batch = X_batch.to(device), y_batch.to(device).type(torch.cuda.FloatTensor)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # 위에서 torch.cuda.FloatTensor 해주는 이유?ㅊ

        optimizer.zero_grad()
        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch)
        _, predicted = torch.max(y_pred.data, 1)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += (predicted == y_batch).sum().item()
        output = y_pred.cpu()
        label = y_batch.cpu()
        _, preds = torch.max(output, dim=1)
        epoch_f1 += f1_score(label, preds, average='weighted')

    # validate-the-model
    model.eval()
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        val_loss += loss.item()

    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    },
        f"saved/0826_{e}_{epoch_loss / len(train_loader):.5f}_{val_loss / len(train_loader):.5f}_{epoch_acc / len(train_loader):.5f}_{epoch_f1 / len(train_loader):.3f}.pt")

    print(
        f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f}| val_loss:{val_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f} | F1: {epoch_f1 / len(train_loader):.3f}')
#     print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

# TEST
class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

import pandas as pd
# meta 데이터와 이미지 경로를 불러옵니다.
submission = pd.read_csv(os.path.join(test_base_path, 'info.csv'))
image_dir = os.path.join(test_base_path, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
#     Resize((512, 384), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.56019358, 0.52410121, 0.501457), std=(0.23318603, 0.24300033, 0.24567522)),
])
test_dataset = TestDataset(image_paths, transform)

test_loader = DataLoader(
    test_dataset,
    shuffle=False
)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device('cuda')

model = MyModel(num_classes).to(device)
# model = torch.nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# checkpoint = torch.load("saved/0826_88_0.48276_0.16753_52.50633_0.815.pt")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
model.load_state_dict(torch.load('/opt/ml/baselinecode/model/exp30/best.pth'))
model.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for images in test_loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다. # 시간으로 저장해서 submission이 겹치지 않게 해줌
from pytz import timezone
import datetime as dt
# 제출할 파일을 저장합니다.
now = (dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S"))
submission.to_csv(f"/opt/ml/input/data/eval/sub_efficientnet_{now}.csv", index=False)
print('test inference is done!')