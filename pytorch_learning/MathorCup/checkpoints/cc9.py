import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
im=torch.ones(1,3,7,7)
net=nn.Sequential(
    nn.Conv2d(3,2,5,padding=2)
)
cc=net[0].bias
im2=net(im)
a=3