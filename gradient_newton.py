import time
import numpy as np
import matplotlib.pyplot as plt
import conjugate_gradient_method as cgm

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
    start = time.clock()

    """
    方向修正的牛顿法
    """
    # 存储迭代点
    W=np.zeros((2,10**3))
    iter=0
    x=x0
    # 使用梯度法计算初始值 算法的收敛条件需要比newton法弱
    x = gradient_backtracking(np.array([0,0]), 2, 0.5, 0.5, 1e-2)
    gval=jacobian(x)
    hval=hessian(x)
    # 判断hessian矩阵是否正定
    try:
        # 正定为newton方向
        L = np.linalg.cholesky(hval)
        # 直接求逆求解newton方向
        #d = np.dot(np.linalg.inv(hessian(x)),jacobian(x))
        # 使用cholesky分解求解newton方向
        d = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L),gval)) 
        # 使用共轭梯度法求解newton方向
        #d = cgm.cg(hval, gval, np.zeros(len(gval)), 1e-8, 100)
    except np.linalg.LinAlgError:
        # 非正定选用梯度方向进行迭代
        print("iter={iter}, hessian is not positive definite".format(iter=iter+1))
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
            # 正定为newton方向
            L = np.linalg.cholesky(hval)
            # 直接求逆求解newton方向
            #d = np.dot(np.linalg.inv(hessian(x)),jacobian(x))
            # 使用cholesky分解求解newton方向
            d = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L),gval)) 
            # 使用共轭梯度法求解newton方向
            #d = cgm.cg(hval, gval, np.zeros(len(gval)), 1e-8, 100) 
        except np.linalg.LinAlgError:
            # 非正定选用梯度方向进行迭代
            print("iter={iter}, hessian is not positive definite".format(iter=iter+1))
            d = gval    

        print("iter point is {x}".format(x=x))     

    end = time.clock()
    
    print("newton_hybrid method time cost is {cost}".format(cost=(end-start)*1000))
    if iter==10000:
        print('did not converge')
      
    W=W[:,0:iter-1] 
    return x, iter, W
          
def gradient_backtracking(x0,s,alpha,beta, epsilon):
    """
    基于回溯法的梯度法
    x0: 初始点
    s: 初始步长
    alpha: 步长选择的公差参数
    beta: 每一步回溯时步长乘以的常数(0 < beta < 1)
    epsilon: 终止准则
    return: x 
    """
    x=x0
    # 计算梯度
    grad=jacobian(x)
    fun_val=fval(x)
    iter=0
    while np.linalg.norm(grad) > epsilon:
        iter=iter+1
        t=s
        while fun_val-fval(x-t*grad) < alpha*t*np.linalg.norm(grad)**2:
            t = beta *t

        x=x-t*grad
        fun_val=fval(x)
        grad=jacobian(x)
        
        # print('iter_number = {iter} norm_grad = {norm_grad:2.6f} fun_val = {fval:2.6f}'.format(
        #     iter=iter,norm_grad=np.linalg.norm(grad),fval=fun_val))      

    return x    

if __name__=="__main__":
    x0 = np.array([10,10])           
    alpha = 0.5
    beta = 0.5
    epsilon = 1e-5 
    iter, x, W = newton_hybrid(x0, alpha, beta, epsilon)
    # print("interation point is {W0}".format(W0=W[0,:]))
    # print("interation point is {W1}".format(W1=W[1,:]))
    # plt.plot(W[0,:],W[1,:],'g*',W[0,:],W[1,:]) # 画出迭代点收敛的轨迹
    # plt.show() # 显示轨迹

    # 梯度法的测试
    # x0 = np.array([0,0])
    # s = 2
    # alpha = 0.5
    # beta = 0.5
    # epsilon = 1e-2
    # x = gradient_backtracking(x0,s,alpha,beta, epsilon)
    # print("solution is {x}".format(x=x))