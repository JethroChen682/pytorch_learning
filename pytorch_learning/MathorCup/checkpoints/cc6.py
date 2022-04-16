import numpy as np
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据

from torch import nn
from torch.autograd import Variable
# 使用内置函数下载 mnist 数据集
device = torch.device('cuda:0')
d=torch.device("cpu")
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,)) # 拉平
    x = torch.Tensor(x)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=False) # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=False)
from torch.utils.data import DataLoader
# 使用 pytorch 自带的 DataLoader 定义一个数据迭代器
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)
class module_net(nn.Module):
    def __init__(self):
        super(module_net, self).__init__()
        self.l1=nn.Linear(28*28,400)
        self.l15=nn.ReLU()
        self.l2=nn.Linear(400,200)
        self.l25=nn.ReLU()
        self.l3=nn.Linear(200,100)
        self.l35=nn.ReLU()
        self.l4 = nn.Linear(100, 10)

    def forward(self,x):
        x=self.l1(x)
        x=self.l15(x)
        x=self.l2(x)
        x=self.l25(x)
        x=self.l3(x)
        x=self.l35(x)
        x=self.l4(x)
        return x
net=module_net().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),1e-1)
losses = []
acces = []
eval_losses = []
eval_acces = []


for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im).to(device)

        label = Variable(label).to(device)
        # 前向传播
        out = net(im).to(device)

        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.data
        # 计算分类的准确率
        _, pred= out.max(1)
        num_correct = (pred == label).sum().data
        acc = num_correct / im.shape[0]
        train_acc += acc
    cc=len(train_data)
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
a=1
