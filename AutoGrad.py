import torch
import torch.nn as nn
import torch.nn.functional as F

# autograd
x = torch.randn(2,requires_grad=True)
y = x * 3
gradients = torch.tensor([100,0.1], dtype=torch.float)
y.backward(gradients, retain_graph=True)
y.backward(gradients)
print(x.grad)

# grad_fn
x = torch.randn(2,requires_grad=True)
y = x*3
z = x/2
w = x+y
print(w,y,z)

# hook
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,20,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(320, 50)
        self.out = nn.Linear(50,10)

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = F.relu(self.out(x))
        return x

    def hook_func(self, input, output):
        print('inside'+self.__class__.__name__ + 'forward')
        print('')
        print('input:',type(input))
        print('input[0]:',type(input[0]))
        print('output', type(output))
        print('')

    def hook_pre(self, input):
        print('inside' + self.__class__.__name__+'forward')
        print('')
        print('input:', type(input))
        print('input[0]:', type(input[0]))
        print('')

    def hook_grad(self, grad_input, grad_output):
        print('inside' + self.__class__.__name__ + 'backward')
        print('')
        print('input:', type(grad_input))
        print('input[0]:', type(grad_input[0]))
        print('output', type(grad_output))
        print('')

net = SimpleNet()
net.conv1.register_forward_hook(net.hook_func)
net.conv2.register_forward_hook(net.hook_func)

input = torch.randn(1,1,28,28)
out = net(input)

# pre_hook
net.conv1.register_forward_pre_hook(net.hook_pre)

# backward_hook
net.conv1.register_backward_hook(net.hook_grad)

target = torch.tensor([3], dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()
err = loss_fn(out, target)
err.backward()

# register 지우는 법
h = net.conv1.register_forward_hook(net.hook_func)
h.remove()
out = net(input)

# hook으로 activation map 가져오기
save_feat = []
def hook_feat(module, input, output):
    save_feat.append(output)
    return output

for name, module in net.get_model_shortcuts():
    if (name == 'target_layer_name'):
        net.register_forward_hook(hook_feat)

img = img.unsqueeze(0)
s = net(img)[0]