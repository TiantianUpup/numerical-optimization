from cgi import print_environ_usage
from turtle import dot
import numpy as np

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

def newton_hybrid(x0,alpha,beta,epsilon):
    iter=0
    x=x0
    gval=jacobian(x)
    hval=hessian(x)
    # 判断hessian矩阵是否正定
    try:
        np.linalg.cholesky(hval)
        # 正定为newton方向
        d = np.dot(np.linalg.inv(hessian(x)),jacobian(x)) 
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
        print('iter={iter} f(x)={fval:10.10f}'.format(iter=iter,fval=fval(x)))
        gval=jacobian(x)
        hval=hessian(x)
        
        try:
            np.linalg.cholesky(hval)
            # 正定为newton方向
            d = np.dot(np.linalg.inv(hessian(x)),jacobian(x)) 
        except np.linalg.LinAlgError:
            # 非正定选用梯度方向进行迭代
            print("iter={iter}, hessian is not positive definite".format(iter=iter+1))
            d = gval     

    if iter==10000:
        print('did not converge')

    return x, iter    

if __name__=="__main__":
    x0 = np.array([2,5])           
    alpha = 0.5
    beta = 0.5
    epsilon = 1e-5 
    iter, x = newton_hybrid(x0, alpha, beta, epsilon)