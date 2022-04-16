import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
im=torch.ones(3,1,3,3)
conv1=nn.Conv2d(3,10,3,bias=False)

sobel_kernel = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,0],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])# 定义轮廓检测算子
sobel_kernel=sobel_kernel.astype("float32")
sobel_kernel = sobel_kernel.reshape((3, 1, 3, 3)) # 适配卷积的输入输出
conv1.weight.data = torch.Tensor(sobel_kernel) # 给卷积的 kernel 赋值

edge1 = conv1(im) # 作用在图片上
edge1 = edge1.data.squeeze().numpy() # 将输出转换为图片的格式
a=3
