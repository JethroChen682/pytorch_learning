import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
d=torch.device("cuda:0")
x=abs(torch.randn(100000,1))
x.to(d)
y=x**4+x**2
y.to(d)
net=nn.Sequential(
    nn.Linear(1,20),
    nn.ReLU(),
    nn.Linear(20, 40),
    nn.Tanh(),
    nn.Linear(40, 20),
    nn.ReLU(),
    nn.Linear(20,1)
)

cri=torch.nn.MSELoss()
opt=torch.optim.Adam(net.parameters())
net.train()
net.to(d)
for e in range(1,1000):
    x,y=x.to(d),y.to(d)
    y_=net(x).to(d)
    loss=cri(y_,y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if e%20==0:
        print(f"e:{e} loss:{loss.data}\n")
torch.save(net,r"nnnnnn1.pth")