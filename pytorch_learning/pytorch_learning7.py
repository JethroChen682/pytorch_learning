import torch
from torch.autograd import Variable
data=[[1,2],[5,7]]
tensor=torch.FloatTensor(data)
var=Variable(tensor,requires_grad=True)
mm=torch.mm(var,var)
t_out=torch.mean(tensor*tensor)
v_out=torch.mean(var*var)
v_out.backward()
#v_out=(1/4)*(var**2)
var.grad#var.grad=d(v_out)/d(var)=0.5*var