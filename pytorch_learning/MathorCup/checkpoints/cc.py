import torch as to
from torch.autograd import Variable
lr=[0.1]
lr=to.Tensor(lr)
x=to.randn(20000)
y=to.randn(20000)

a=Variable(to.Tensor(1),requires_grad=True)
b=Variable(to.Tensor(1),requires_grad=True)
for i in range(20000):
    loss=(y.data[i]-x.data[i]*a-b)**2
    if i!=0:
        a.grad.zero_()
        b.grad.zero_()
    loss.backward()
    a.data=a.data-a.grad*0.001
    b.data=b.data-b.grad*0.001
print(f"{a.data}\n{b.data}")
