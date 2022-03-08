import numpy as np
import matplotlib.pyplot as plt

def jacobian(x):
    return  np.array([-400*(x[1]-x[0]**2)*x[0]-2*(1-x[0]),200*(x[1]-x[0]**2)],dtype=np.float64)

"""
计算hessian矩阵
x: 迭代点
"""
def hessian(x):
    return np.array([
        [-400*x[1]+1200*x[0]**2+2,-400*x[0]],
        [-400*x[0] ,200] ])

"""
计算f的函数值
"""
def fval(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

X1=np.arange(-3,3+0.05,0.05)
X2=np.arange(-2,8+0.05,0.05)
[x1,x2]=np.meshgrid(X1,X2)
f=100*(x2-x1**2)**2+(1-x1)**2
plt.contour(x1,x2,f,20) # 绘制20条等值线

def newton_hybrid(x0,alpha,beta,epsilon):
    # 存储迭代点
    W=np.zeros((2,10**3))
    iter=0
    x=x0
    gval=jacobian(x)
    hval=hessian(x)
    # 判断hessian矩阵是否正定
    try:
        L = np.linalg.cholesky(hval)
        # 正定为newton方向
        #d = np.dot(np.linalg.inv(hessian(x)),jacobian(x))
        d = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L),jacobian(x))) 
    except np.linalg.LinAlgError:
        # 非正定选用梯度方向进行迭代
        # print("iter={iter}, hessian is not positive definite".format(iter=iter+1))
        d = gval     

    while np.linalg.norm(gval)>epsilon and iter<10000:
        iter=iter+1
        t=1
        while(fval(x-t*d)>fval(x)-alpha*t*np.dot(gval, d)):
            t=beta*t
        
        x=x-t*d
        W[:,iter-1] = x
        print('iter={iter} f(x)={fval:10.10f}'.format(iter=iter,fval=fval(x)))
        gval=jacobian(x)
        hval=hessian(x)
        
        try:
            L = np.linalg.cholesky(hval)
            # 正定为newton方向
            #d = np.dot(np.linalg.inv(hessian(x)),jacobian(x))
            d = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L),jacobian(x)))  
        except np.linalg.LinAlgError:
            # 非正定选用梯度方向进行迭代
            print("iter={iter}, hessian is not positive definite".format(iter=iter+1))
            d = gval    

        print("iter point is {x}".format(x=x))     

    if iter==10000:
        print('did not converge')
      
    W=W[:,0:iter-1] 
    return x, iter, W    

if __name__=="__main__":
    x0 = np.array([2,5])           
    alpha = 0.5
    beta = 0.5
    epsilon = 1e-5 
    iter, x, W = newton_hybrid(x0, alpha, beta, epsilon)
    print("interation point is {W0}".format(W0=W[0,:]))
    print("interation point is {W1}".format(W1=W[1,:]))
    plt.plot(W[0,:],W[1,:],'g*',W[0,:],W[1,:]) # 画出迭代点收敛的轨迹
    plt.show() # 显示轨迹