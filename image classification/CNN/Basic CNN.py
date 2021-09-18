import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# DATASET
mnist_train = datasets.MNIST(root = './data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor(),download=True)
print ("mnist_train:\n",mnist_train,"\n")
print ("mnist_test:\n",mnist_test,"\n")

BATCH_SIZE = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

# Model
class CNN(nn.Module):
    def __init__(self, name = 'cnn',
                 xdim=[1,28,28], ksize = 3, cdims = [32,64], hdims = [1024,128], ydim = 10,
                 USE_BATCHNORM = False):
        super(CNN, self).__init__()
        self.name = name
        self.xdim = xdim
        self.ksize = ksize
        self.cdims = cdims
        self.hdims = hdims
        self.ydim = ydim
        self.USE_BATCHNORM = USE_BATCHNORM

        self.layers = [] # layer 리스트
        prev_cdim = self.xdim[0] # channel dimension (흑백)
        for cdim in self.cdims:
            self.layers.append(
                # nput spacial dimension이 output과 똑같게 하기 위해
                # (filter 3*3 + padding 1, filter 5*5 + padding 2)
                nn.Conv2d(
                    in_channels = prev_cdim,
                    out_channels=cdim,
                    kernel_size=self.ksize,
                    stride=(1,1),
                    padding=self.ksize//2
               )
            )
            # batch norm
            if self.USE_BATCHNORM:
                self.layers.append(nn.BatchNorm2d(cdim))
            # activation
            self.layers.append(nn.ReLU(True))
            #max-pooling
            self.layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
            #Dropout
            self.layers.append(nn.Dropout2d(p=0.5))
            prev_cdim = cdim

        # Dense Layer
        self.layers.append(nn.Flatten())
        prev_hdim = prev_cdim * (self.xdim[1]//(2**len(self.cdims))) * (self.xdim[2]//2**len(self.cdims))
        for hdim in self.hdims:
            self.layers.append(nn.Linear(
                prev_hdim, hdim, bias = True
            ))
            self.layers.append(nn.ReLU(True))
            prev_hdim = hdim
        self.layers.append(nn.Linear(prev_hdim, self.ydim, bias=True))

        self.net = nn.Sequential()
        # Pytorch에서 layer 이름 지정하는 방법
        for I_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(), I_idx)
            self.net.add_module(layer_name, layer)
        self.init_param() # initialize parameter
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight) # He initialization(Resnet 기반)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):  # init BN
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

C = CNN(
    name='cnn',xdim=[1,28,28],ksize=3,cdims=[32,64], # 32*32 ->14*14->7*7
    hdims=[32],ydim=10).to(device)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(C.parameters(),lr=1e-3)

# Check Parameters
np.set_printoptions(precision=3)
n_param = 0
for p_idx,(param_name,param) in enumerate(C.named_parameters()):
    if param.requires_grad:
        param_numpy = param.detach().cpu().numpy() # to numpy array
        n_param += len(param_numpy.reshape(-1))
        print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
        print ("    val:%s"%(param_numpy.reshape(-1)[:5]))
print ("Total number of parameters:[%s]."%(format(n_param,',d')))

'''
[0] name:[net.conv2d_00.weight] shape:[(32, 1, 3, 3)].
    val:[ 0.031  0.008  0.049 -0.253 -0.308]
[1] name:[net.conv2d_00.bias] shape:[(32,)].
    val:[0. 0. 0. 0. 0.]
[2] name:[net.conv2d_04.weight] shape:[(64, 32, 3, 3)].
    val:[ 0.153 -0.127  0.115  0.051  0.138]
[3] name:[net.conv2d_04.bias] shape:[(64,)].
    val:[0. 0. 0. 0. 0.]
[4] name:[net.linear_09.weight] shape:[(32, 3136)].
    val:[-0.007 -0.003  0.033 -0.014 -0.024]
[5] name:[net.linear_09.bias] shape:[(32,)].
    val:[0. 0. 0. 0. 0.]
[6] name:[net.linear_11.weight] shape:[(10, 32)].
    val:[-0.23  -0.164  0.131  0.107  0.257]
[7] name:[net.linear_11.bias] shape:[(10,)].
    val:[0. 0. 0. 0. 0.]
Total number of parameters:[119,530].
'''

# Simple Forward Path of the CNN Model
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)
x_numpy = np.random.rand(2,1,28,28) # batch 가 2개, gray scale image, 28*28
x_torch = torch.from_numpy(x_numpy).float().to(device)
y_torch = C.forward(x_torch) # forward path
y_numpy = y_torch.detach().cpu().numpy() # torch tensor to numpy array
print ("x_torch:\n",x_torch)
print ("y_torch:\n",y_torch)
print ("\nx_numpy %s:\n"%(x_numpy.shape,),x_numpy)
print ("y_numpy %s:\n"%(y_numpy.shape,),y_numpy)

# Evaluation Function
def func_eval(model,data_iter,device):
    with torch.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.view(-1,1,28,28).to(device))
            _,y_pred = torch.max(model_pred.data,1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode
    return val_accr


# Initial Evaluation
C.init_param() # initialize parameters
train_accr = func_eval(C,train_iter,device)
test_accr = func_eval(C,test_iter,device)
print ("train_accr:[%.3f] test_accr:[%.3f]."%(train_accr,test_accr))
# 여기서는 모든 값을 초기화 했기 때문에 라벨이 10개이니 대충 10%의 정확도가 나올 것이다

# 앞의 mlp와 다른 점이 없음
print ("Start training.")
C.init_param() # initialize parameters
C.train() # to train mode
EPOCHS,print_every = 10,1
for epoch in range(EPOCHS):
    loss_val_sum = 0
    for batch_in,batch_out in train_iter:
        # Forward path
        y_pred = C.forward(batch_in.view(-1,1,28,28).to(device))
        loss_out = loss(y_pred,batch_out.to(device))
        # Update
        optm.zero_grad()      # reset gradient
        loss_out.backward()      # backpropagate
        optm.step()      # optimizer update
        loss_val_sum += loss_out
    loss_val_avg = loss_val_sum/len(train_iter)
    # Print
    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        train_accr = func_eval(C,train_iter,device)
        test_accr = func_eval(C,test_iter,device)
        print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
               (epoch,loss_val_avg,train_accr,test_accr)) # 매우 간단한 CNN이어도 MLP보다 일반화 성능이 좋다는 것을 알 수 있음
print ("Done")

n_sample = 25
sample_indices = np.random.choice(len(mnist_test.targets),n_sample,replace=False)
test_x = mnist_test.data[sample_indices]
test_y = mnist_test.targets[sample_indices]
with torch.no_grad():
    C.eval() # to evaluation mode
    y_pred = C.forward(test_x.view(-1,1,28,28).type(torch.float).to(device)/255.)
y_pred = y_pred.argmax(axis=1)
plt.figure(figsize=(10,10))
for idx in range(n_sample):
    plt.subplot(5, 5, idx+1)
    plt.imshow(test_x[idx], cmap='gray')
    plt.axis('off')
    plt.title("Pred:%d, Label:%d"%(y_pred[idx],test_y[idx]))
plt.show()
print ("Done")