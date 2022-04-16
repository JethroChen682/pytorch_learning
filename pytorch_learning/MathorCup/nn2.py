import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
d=torch.device("cuda:0")
label=torch.Tensor(100000)
for  i in range(10):
    label[i*10000:(i+1)*10000]=torch.Tensor([i])
label=label.long()
xxx=label.float()**3+label.float()**2+torch.randn(100000)
xxx=xxx.view(-1,1)
net=nn.Sequential(
    nn.Linear(1,20),
    nn.ReLU(),
    nn.Linear(20, 40),
    nn.LeakyReLU(),
    nn.Linear(40, 20),
    nn.ReLU(),
    nn.Linear(20,10),
    nn.Softmax(dim=1)
)
net.to(d)
cri=nn.CrossEntropyLoss()
opt=torch.optim.Adam(net.parameters())

net.train()

for e in range(1,1000):
    xxx,label=xxx.to(d),label.to(d)
    y_=net(xxx).to(d)
    loss=cri(y_,label).to(d)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if e%20==0:
        print(f"e:{e} loss:{loss}\n")
torch.save(net,r"nn2.pth")
