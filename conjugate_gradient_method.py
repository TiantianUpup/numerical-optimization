import numpy as np
import math

def cg(A, b, x, eta, i_max):
    """共轭梯度法求解方程Ax=b，这里A为对称正定矩阵
    Args:
        A: 方程系数矩阵
        b:
        x: 初始迭代点
        eta: 收敛条件
        i_max: 最大迭代次数
    Returns: 方程的解x
    """
    i = 0 # 迭代次数
    r_0 = np.dot(A, x)-b
    p = -r_0
    r_norm = np.linalg.norm(r_0)
    while r_norm > eta and i < i_max:
        alpha = np.dot(r_0,r_0) / np.dot(p, np.dot(A, p))
        x = x + alpha * p
        r_1 = r_0 + alpha * np.dot(A, p)
        
        beta = np.dot(r_1, r_1) / np.dot(r_0, r_0)
        p = -r_1 + beta * p
        r_0 = r_1
        r_norm = np.linalg.norm(r_0)
        i = i + 1
   
    return x     

def semiPosiM(low, high, size):
    A = np.random.uniform(low, high, size=(size, size)) # 产生一个随机整数矩阵
    B = np.dot(A, A.T)
    return B    

if __name__=="__main__":
    # A = np.array([
    #     [1,0,0],
    #     [0,2,0],
    #     [0,0,4]
    # ])
    size = 5

    # 必须保证A为正定矩阵
    # A =  semiPosiM(-1,2,size)
    # print("matrix A is:{A}\n".format(A = A))
    # xtemp = np.random.rand(size)
    # print("theoretical solution is:{x}".format(x = xtemp))
    # b = A.dot(xtemp)
    # print("b is:{b}".format(b = b))
    # x = np.zeros(size)
    # eta = 1e-8
    # i_max = 100
    A = np.array([[4,1],
        [1,3]])  
    b = np.array([1,2])
    
    x = np.array([2,1])
    eta = 1e-8
    i_max = 100
    x,i = cg(A, b, x, eta, i_max)
    print("algotirhm solution is:{x}, iterater is:{i}".format(x = x, i = i))