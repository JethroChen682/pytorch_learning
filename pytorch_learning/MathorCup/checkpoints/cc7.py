import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
im=Image.open("./cat.png").convert("L")
im=np.array(im,"float32")
# plt.figure(num=1)
# plt.imshow(im.astype("uint8"),cmap="gray")
# plt.figure(num=2)
# plt.imshow(im,cmap="rainbow")
im=torch.Tensor(im.reshape(1,1,im.shape[0],im.shape[1]))
# 使用 nn.Conv2d
conv1 = nn.Conv2d(1, 1, 3, bias=False) # 定义卷积

sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') # 定义轮廓检测算子
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3)) # 适配卷积的输入输出
conv1.weight.data = torch.from_numpy(sobel_kernel) # 给卷积的 kernel 赋值

edge1 = conv1(im) # 作用在图片上
edge1 = edge1.data.squeeze().numpy() # 将输出转换为图片的格式
# 使用 F.conv2d
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') # 定义轮廓检测算子
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3)) # 适配卷积的输入输出
weight = Variable(torch.from_numpy(sobel_kernel))

edge2 = F.conv2d(im, weight) # 作用在图片上
edge2 = edge2.data.squeeze().numpy() # 将输出转换为图片的格式
# plt.imshow(edge2, cmap='gray')
# plt.show()
# pool1 = nn.MaxPool2d(2, 2)
# print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
# small_im1 = pool1(Variable(im))
# small_im1 = small_im1.data.squeeze().numpy()
# print('after max pool, image shape: {} x {} '.format(small_im1.shape[0], small_im1.shape[1]))
print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
small_im2 = F.max_pool2d(Variable(im), 2, 2)
small_im2 = small_im2.data.squeeze().numpy()
print('after max pool, image shape: {} x {} '.format(small_im2.shape[0], small_im2.shape[1]))
plt.imshow(small_im2, cmap='gray')
plt.show()
a=3