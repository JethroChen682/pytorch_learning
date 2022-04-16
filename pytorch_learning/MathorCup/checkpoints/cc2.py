import torch as to
from torch.autograd import Variable
w=Variable(to.randn(2,1),requires_grad=True)
b=Variable(to.zeros(1),requires_grad=True)
x=to.randn(200,2)
y=to.rand(200,1)
def logistic(x):
    y_=to.sigmoid(x.mm(w)+b)
    return y_
def loss(y,y_):
    loss =