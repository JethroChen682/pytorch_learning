import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
device = torch.device('cuda:0')
d=torch.device('cpu')
def plot_decision_boundary(model, x, y):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #把坐标里的每一点带入得到z,然后画出等值线
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色
a = 4

for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=(40,), cmap=plt.cm.Spectral)


class module_net(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(module_net, self).__init__()
        self.dog = nn.Linear(num_input, num_hidden)

        self.layer2 = nn.Tanh()

        self.layer3 = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.dog(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
seq_net=module_net(2,4,1)
param=seq_net.parameters()
optimizer=torch.optim.SGD(param,1.)
x=torch.Tensor(x).to(device)
y=torch.Tensor(y).to(device)
criterion = nn.BCEWithLogitsLoss()
seq_net.to(device)
for e in range(100000):
    out=seq_net(x).to(device)
    loss=criterion(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#y=x.mm(w.t)+b
torch.save(seq_net,r"./abc/ad.txt")
mnet=torch.load(r"./abc/ad.txt")
net=seq_net


def plot_net(x):
    out = torch.sigmoid(net(Variable(torch.from_numpy(x).float()).to(device))).to(d).data.numpy()
    out = (out > 0.5) * 1
    return out

plot_decision_boundary(lambda x: plot_net(x), x.to(d).numpy(), y.to(d).numpy())
plt.title('sequential')
plt.show()