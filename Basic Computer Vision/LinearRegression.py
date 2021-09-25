import torch
import torch.nn as nn
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # weight의 개수 = input_channel * output_channel
        self.weights = nn.Parameter(
            torch.randn(in_features, out_features)
        )
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weights + self.bias

x = torch.randn(5,7)
layer = MyLinear(7,12)
layer(x).shape # torch.Size([5, 12])

for value in layer.parameters():
    print(value)

import numpy as np
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
# print(x_train.shape) # (11,)
x_train = x_train.reshape(-1,1)
# print(x_train.shape) # (11, 1)

y_values = [2*i+1 for i in x_values]
y_train = np.array(y_values, dtype = np.float32)
y_train = y_train.reshape(-1,1)

from torch.autograd import Variable
class LinearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

inputDim = 1
outputDim = 1

model = LinearRegression(inputDim, outputDim)

if torch.cuda.is_available():
    model.cuda()

lr = 0.01
epochs = 100
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

with torch.no_grad():
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

for p in model.parameters():
    if p.requires_grad:
        print(p.name, p.data)