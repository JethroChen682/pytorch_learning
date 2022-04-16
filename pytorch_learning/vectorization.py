import numpy as np
import time
import sklearn
m=1000000
m=int(m)
a=np.random.rand(m)
b=np.random.rand(m)
tic=time.time()
c=np.dot(a,b)
toc=time.time()
print(c)
print("vectorized version:"+str((10^3)*(toc-tic))+"ms")
c=0
tic=time.time()
for i in range(m):
    c+=a[i]*b[i]
toc=time.time()
print(c)
print("for loop:"+str(1000*(toc-tic))+"ms")
