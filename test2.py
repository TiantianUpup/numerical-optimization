import numpy as np
import matplotlib.pyplot as plt

X1=np.arange(-3,3+0.05,0.05)
X2=np.arange(-2,5+0.05,0.05)
[x1,x2]=np.meshgrid(X1,X2)
f=100*(x2-x1**2)**2+(1-x1)**2
plt.contour(x1,x2,f,20)
plt.show()