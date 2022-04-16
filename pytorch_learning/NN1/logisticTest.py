import numpy as np

nx=10    #自变量的类型数
m=50     #样本数
alpha=0.5#学习率
w=np.random.rand(1,nx).T#自变量的权重
x=np.random.rand(nx,m)#自变量
b=np.random.rand(1)    #截距
y=(np.random.rand(1,m)*2.0).astype(int)  #真实值

z=np.dot(w.T,x)+b
a=1/(1+np.exp(-1*z))   #激活函数导出预测值
loss=-(y*np.log(a)+(1-y)*np.log(1-a))#成本函数
dz=a-y
dw=np.dot(x,dz.T)/m
db=np.sum(dz,axis=1)/m
w=w-alpha*dw
b=b-alpha*db#梯度下降法