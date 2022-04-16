import numpy as np
a=np.array([1,2,3,4,5,6,7,8])
a=a.reshape(2,4)
b=np.array([9,10,11,12,13,14,15,16,17,18,19,20])
b=b.reshape(4,3)
c=np.dot(a,b)
d=np.zeros(6)
d=d.reshape(2,3)
for i in range(2):
    for j in range(3):
        d[i][j]=a[i][0]*b[0][j]+a[i][1]*b[1][j]+a[i][2]*b[2][j]+a[i][3]*b[3][j]
print(c==d)
