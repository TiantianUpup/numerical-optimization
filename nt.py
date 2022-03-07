import numpy as np
"""
梯度的计算
x: 迭代点
"""
def jacobian(x):
    return np.array([-400*(x[1]-x[0]**2)*x[0]-2*(1-x[0]),200*(x[1]-x[0]**2)])

"""
计算hessian矩阵
x: 迭代点
"""
def hessian(x):
    return np.array([
        [-400*x[1]+1200*x[0]**2+2,-400*x[0]],
        [-400*x[0],200]])

"""
计算f的函数值
"""
def fval(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

def newton_hybrid(x0,alpha,beta,epsilon):
    x=x0
    # 当前迭代点的梯度
    gval=jacobian(x)
    # 当前迭代点的hessian矩阵
    #hval=hessian(x)
    # 判断hessian矩阵是否正定
    #d =  d = np.dot(np.linalg.inv(hessian(x)),jacobian(x)) 
    # try:
    #     np.linalg.cholesky(hval)
    #     # 正定为newton方向
    #     d = np.dot(np.linalg.inv(hessian(x)),jacobian(x)) 
    # except np.linalg.LinAlgError:
    #     # 非正定选用梯度方向进行迭代
    #     d = gval     

    iter=0 # 记录迭代次数
    g_norm=np.linalg.norm(jacobian(x))
    while g_norm >epsilon and iter<6:
        iter=iter+1
        d = -np.dot(np.linalg.inv(hessian(x)),jacobian(x))
        t=1
        # 保证充分下降性
        print('===========np.dot(gval,d) is {inner}'.format(inner=np.dot(gval,d)))
        while fval(x+t*d)>fval(x)+alpha*t*np.dot(gval,d):
            t=beta*t

        # x=x-t*d
        
        x = x + t*d
        gval=jacobian(x)
        hval=hessian(x)
        # try:
        #     np.linalg.cholesky(hval)
        #     # 正定为newton方向
        #     d = np.dot(np.linalg.inv(hessian(x)),jacobian(x)) 
        # except np.linalg.LinAlgError:
        #     # 非正定选用梯度方向进行迭代
        #     d = gval     
        g_norm=np.linalg.norm(jacobian(x)) 
        if iter ==1000:
            print('did not converge')
    return iter, x        

if __name__=="__main__":
    x0=np.array([2,5])           
    alpha = 0.5
    beta = 0.5
    epsilon = 1e-5 
    iter,x = newton_hybrid(x0, alpha, beta, epsilon)
    print(x)
    print(iter)
