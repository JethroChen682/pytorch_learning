import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
axe=Axes3D(fig)
x=np.arange(-4,4,0.25)
y=np.arange(-5,5,0.25)
X,Y=np.meshgrid(x,y)
R=np.sqrt(X**2+Y**2)
Z=np.sin(R)
axe.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap("cool"))
plt.show()