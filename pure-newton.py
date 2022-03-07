"""
Newton法
Rosenbrock函数
函数 f(x)=100*(x(2)-x(1).^2).^2+(1-x(1)).^2
梯度 g(x)=(-400*(x(2)-x(1)^2)*x(1)-2*(1-x(1)),200*(x(2)-x(1)^2))^(T)
"""

from stat import FILE_ATTRIBUTE_REPARSE_POINT
import numpy as np
import matplotlib.pyplot as plt

def jacobian(x):
    #return np.array([400*x[0]**3,0.04*x[1]**3])
    # # Example 5.4
    # return np.array([x[0]/np.sqrt(x[0]**2+1),x[1]/np.sqrt(x[1]**2+1)])
    # Example 5.8
    return np.array([-400*(x[1]-x[0]**2)*x[0]-2*(1-x[0]),200*(x[1]-x[0]**2)])

def hessian(x):
    #return np.array([[1200*x[0]**2,0],[0,0.12*x[1]**2]])
    # Example 5.4
    # return np.diag([1/((x[0]**2+1)**1.5),1/((x[1]**2+1)**1.5)])
    # Example 5.8
    return np.array([
        [-400*x[1]+1200*x[0]**2+2,-400*x[0]],
        [-400*x[0],200]])

X1=np.arange(1,6+0.05,0.05)
X2=np.arange(1,6+0.05,0.05)
[x1,x2]=np.meshgrid(X1,X2)
#f=100*x1**4+0.01*x2**4; # 给定的函数
# Example 5.4
f=np.sqrt(1+x1**2)+np.sqrt(1+x2**2)
#plt.contour(x1,x2,f,20) # 画出函数的20条轮廓线


def newton(x0,epsilon):

    W=np.zeros((2,10**3))
    iter = 0
    imax = 1000
    W[:,0] = x0 
    x = x0
    
    # 计算梯度的范数
    g_norm=np.linalg.norm(jacobian(x))
    while iter < 10000 and g_norm > epsilon:
        # 1.计算newton方向
        
        np.set_printoptions(suppress=False)
        d = -np.dot(np.linalg.inv(hessian(x)),jacobian(x)) 
        gval=jacobian(x)
        print('===========np.dot(gval,d) is {inner}'.format(inner=np.dot(gval,d)))
        #d = hessian(x)/jacobian(x)
        # 2.计算迭代点
        x = x + d
        print(x)
        print("f is:{f}".format(f=100*(x[1]-x[0]**2)**2+(1-x[0])**2))
        #W[:,iter] = x
    
        g_norm=np.linalg.norm(jacobian(x))
        iter=iter+1
    #W=W[:,0:iter]  # 记录迭代点
    return iter,x

x0 = np.array([2,5])
iter, x=newton(x0, 1e-5)
print('optimal solution is:{x} '.format(x=x))
print("iter time is:{iter}".format(iter=iter))
print("optimal value is:{f}".format(f=100*(x[1]-x[0]**2)**2+(1-x[0])**2))
# plt.plot(W[0,:],W[1,:],'g*',W[0,:],W[1,:]) # 画出迭代点收敛的轨迹
# plt.show() # 显示轨迹
#print(hessian(np.array([1,1])))
print(hessian(np.array([1,1])))