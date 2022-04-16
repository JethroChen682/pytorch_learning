import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-4,3, 50)
y1 = 2*x+1
y2 = x**2+1


plt.figure(num=3, figsize=(8, 5))
l1, = plt.plot(x, y2)
l2, =plt.plot(x, y1, color="blue", linewidth=2.0, linestyle="--")


plt.xlabel(" i am x")
plt.ylabel(" i am y")
plt.ylim((0,10))
plt.xlim((-4,3))



a=plt.gca()
a.spines["top"].set_color("none")
a.spines["right"].set_color("none")

a.spines["left"].set_position(("data",0))
plt.legend(handles=[l1, l2], labels=["aaa", "bbb"], loc="best")
x0=1
y0=2*x0+1
plt.scatter(x0,y0,color="red")
plt.plot([x0,x0],[y0,0],"y--",lw=2)
plt.annotate("x=1 to this point",(1,3),xycoords="data",xytext=(30,-30)
             ,textcoords="offset points",arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
plt.text(-3.7,5,"complete a graph",fontdict={"size":16,"color":"red"})
plt.show()
