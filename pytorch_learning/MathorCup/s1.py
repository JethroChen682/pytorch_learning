#曲率k=abs(y''')/((1+y'**2)**1.5)
import torch
import numpy as np
#if torch.cuda.is_available():
#    tensor = tensor.to('cuda')
class location:
    def __init__(self,x,y,rad,k,dk,v,a):
        location.co=[x,y]#cooridinate
        location.di=rad#direction,radian
        location.cu=k#curvature
        location.dcu=dk#d(curvature)/dt
        location.v=v#volcity
        location.a=a#accelereated volcity
def update():

    return
def append():

    return
def judgeCrash():

    return
def judgeArchieve():


    return
def init():

    return