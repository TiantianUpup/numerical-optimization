import numpy as np

def jacobian(x):
    return  np.array([x[0]/np.sqrt(x[0]**2+1),x[1]/np.sqrt(x[1]**2+1)])

"""
计算hessian矩阵
x: 迭代点
"""
def hessian(x):
    return np.diag([1/(x[0]**2+1)**(1.5),1/(x[1]**2+1)**(1.5)]) 


"""
计算f的函数值
"""
def fval(x):
    return np.sqrt(1+x[0]**2)+np.sqrt(1+x[1]**2)

def newton_backtracking(x0,alpha,beta,epsilon):
    x=x0
    gval = jacobian(x)
    hval = hessian(x)
    d = np.dot(np.linalg.inv(hessian(x)),jacobian(x))
    iter = 0
   
    while np.linalg.norm(gval) > epsilon and iter < 10000:
        iter = iter+1
        t = 1
        # 添加线性回溯搜索步长
        while(fval(x - t*d) > fval(x) - alpha*t*np.dot(gval,d)):
            t = beta*t

        x = x-t*d
        print("iter= {iter} f(x)={fval:10.10f}".format(iter=iter,fval=fval(x)))
        gval = jacobian(x)
        hval = hessian(x)
        d = np.dot(np.linalg.inv(hessian(x)),jacobian(x))
        if iter == 10000:
            print('did not converge\n')
    return x 

if __name__=="__main__":
    x0 = np.array([10,10])           
    alpha = 0.5
    beta = 0.5
    epsilon = 1e-8 
    iter, x = newton_backtracking(x0, alpha, beta, epsilon)