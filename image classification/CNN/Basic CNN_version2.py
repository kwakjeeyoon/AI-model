import torch
import torch.nn as nn
import torch.optim as optim

import os

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(3*3*64, 1000)
        self.fc2 = nn.Linear(1000,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0),-1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize model
model = TheModelClass()
model.to(device)
# initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# model's state_dict
for param_tensor in model.state_dict():
    print(param_tensor, model.state_dict()[param_tensor].size())
# print(type(model.state_dict())) # <class 'collections.OrderedDict'>

from torchsummary import summary
summary(model, (3,244,244))

# 가중치 저장 및 불러오기
MODEL_PATH = 'saved'
os.makedirs(MODEL_PATH, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'model.pt'))

new_model = TheModelClass()
new_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model.pt')))

# 모델 아키텍쳐 + 가중치 저장 (동일 모델인 경우)
torch.save(model, os.path.join(MODEL_PATH, "model_pickle.pt"))
model = torch.load(os.path.join(MODEL_PATH, "model_pickle.pt"))
model.eval()

from torchvision import datasets
import torchvision.transforms as transforms
import torch

dataset = datasets.ImageFolder(root="data",
                           transform=transforms.Compose([
                               transforms.Scale(244),
                               transforms.CenterCrop(244),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=8,
                                         shuffle=True,
                                         num_workers=8)

import warnings
warnings.filterwarnings("ignore")


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.1

model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).type(torch.cuda.FloatTensor)

        optimizer.zero_grad()
        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, f"saved/checkpoint_model_{e}_{epoch_loss / len(dataloader)}_{epoch_acc / len(dataloader)}.pt")

    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(dataloader):.5f} | Acc: {epoch_acc / len(dataloader):.3f}')