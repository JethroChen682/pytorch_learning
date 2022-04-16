import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
with open('./data.txt', 'r') as f:

    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0]/x0_max, i[1]/x1_max, i[2]) for i in data]
x0 = list(filter(lambda x: x[-1] == 0.0, data)) # 选择第一类的点
x1 = list(filter(lambda x: x[-1] == 1.0, data)) # 选择第二类的点
plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]


np_data = np.array(data, dtype='float32') # 转换成 numpy array
x_data = torch.from_numpy(np_data[:, 0:2]) # 转换成 Tensor, 大小是 [100, 2]
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1) # 转换成 Tensor，大小是 [100, 1]
w = nn.Parameter(torch.randn(2,1))
b = nn.Parameter(torch.zeros(1))


def logistic_regression(x):
    return torch.sigmoid(torch.mm(x, w) + b)
def binary_loss(y_pred, y):
    logits = (y * y_pred.clamp(1e-12).log() + (1 - y) * (1 - y_pred).clamp(1e-12).log()).mean()
    return -logits
optimizer = torch.optim.SGD([w, b], lr=1.)
def logistic_reg(x):
    return torch.mm(x, w) + b
cri=torch.nn.BCEWithLogitsLoss()
torch.nn.Conv3d
for e in range(1000):
    y_=logistic_reg(x_data)
    loss=torch.nn.BCEWithLogitsLoss()(y_,y_data)
    optimizer.zero_grad()  # 使用优化器将梯度归 0
    loss.backward()
    optimizer.step()  # 使用优化器来更新参数
    # 计算正确率

    c=e+1
    if c%20==0:
        mask = y_.ge(0.5).float()
        cc=mask == y_data
        acc = (mask == y_data).sum().data / y_data.shape[0]
        print(f"epoch:{c}+rate:{acc}\n")
w0 = w[0].data[0]
w1 = w[1].data[0]
b0 = b.data[0]

plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b0) / w1

plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')
plt.show()