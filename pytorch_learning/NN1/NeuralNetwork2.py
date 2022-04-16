import numpy as np
m=50
nx=3
node1=4
node2=1
x=np.random.rand(nx,m)
w1=np.random.rand(node1,nx)
w2=np.random.rand(node2,node1)
b1=np.random.rand(node1,1)
b2=np.random.rand(node2,1)
z1=np.dot(w1,x)+b1
a1=(np.exp(z1)-np.exp(-1*z1))/(np.exp(z1)+np.exp(-1*z1))#tanh
da1=1-(a1)**2
# da1/dz1=1-(a1)**2  tanh's derivative
z2=np.dot(w2,a1)+b2
a2=1/(1+np.exp(-1*z2))#sigmoid
da2=a2*(1-a2)
# da2/dz2=a2*(1-a2) sigmoid's derivative