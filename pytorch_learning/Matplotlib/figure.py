import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(0,2*np.pi,100)
y=(1-np.sin(x))*3

ax=plt.subplot(projection="polar")
ax.plot(x,y)
plt.show()