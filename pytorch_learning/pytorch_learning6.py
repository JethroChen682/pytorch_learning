import numpy as np
import torch
data=[-1,-2,3,4]
tensor=torch.FloatTensor(data)
add=torch.add(tensor,30)
abs=torch.abs(tensor)
sin=torch.sin(tensor)
mean=torch.mean(tensor)
ndarray=np.array(data).reshape(2,2)
array_tensor=torch.FloatTensor(ndarray)
array_tensor2=torch.tensor(ndarray)
mm_np=np.matmul(ndarray,ndarray)
mm_tensor=torch.mm(array_tensor,array_tensor)
array_tensor=torch.arange(6)


dot_np=ndarray.dot(ndarray)
dot_tensor=array_tensor.dot(array_tensor)
