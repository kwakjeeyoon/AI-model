# Transfer Learning

```python
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
```

- VGG fine-tuning
```python
class MyNewNet(nn.Module):
    def __init__(self):
        super(MyNewNet, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        self.linear_layers = nn.Linear(1000, 1)


    # Defining the forward pass
    def forward(self, x):
        x = self.vgg19(x)
        return self.linear_layers(x)

```
기존 파라미터 freeze
```python
my_model = MyNewNet()
my_model = my_model.to(device)

for param in my_model.parameters():
    param.requires_grad = False
for param in my_model.linear_layers.parameters():
    param.requires_grad = True

```