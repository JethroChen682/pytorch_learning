import torch
import numpy as np
def distance(a,b,dot):
    a=np.array(a)
    b=np.array(b)
    dot=np.array(dot)
    m=np.vstack((b - a, dot - a))
    distance=abs(np.linalg.det(m))/np.linalg.norm(b-a)


    return distance
if(__name__=="__main__"):
    a=distance([0,1],[0,0],[1,0])
    print(a)