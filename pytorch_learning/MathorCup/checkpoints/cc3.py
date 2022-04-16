import numpy as np
import torch as to
from torch.autograd import Variable
import matplotlib.pyplot as plt


w_target = np.array([0.5, 3, 2.4]) # 定义参数
b_target = np.array([0.9]) # 定义参数
x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3
cc=[x_sample ** i for i in range(1, 4)]
x_train = np.stack([x_sample ** i for i in range(1, 4)],axis=1)
x_train = to.Tensor(x_train)

y_train = to.Tensor(y_sample).unsqueeze(1) # 转化成 float tensor
w = Variable(to.randn(3, 1), requires_grad=True)
b = Variable(to.zeros(1), requires_grad=True)
def get_loss(y_, y):
    return to.mean((y_ - y) ** 2)
def multi_linear(x):
    return to.mm(x, w) + b


# 进行 100 次参数更新
for e in range(100):
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred, y_train)
    if e!=0:
        w.grad.data.zero_()
        b.grad.data.zero_()
    loss.backward()

    # 更新参数
    w.data = w.data - 0.001 * w.grad.data
    b.data = b.data - 0.001 * b.grad.data
    if (e + 1) % 20 == 0:
        print('epoch {}, Loss: {:.5f}'.format(e + 1, loss))
        plt.figure(num=(e + 1) /20)
        plt.plot(x_sample,y_sample,"r",label="real")
        plt.plot(x_sample,y_pred.data.numpy(),"b",label="estimate")
        plt.legend()
plt.show()