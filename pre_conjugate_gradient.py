import numpy as np
import time
from scipy import sparse
from scipy.sparse import linalg

def ilucg(A, B, X, eta, i_max):
    """共轭梯度法求解方程AX=B，这里A为对称正定矩阵
    Args:
        A: 方程系数矩阵
        B:
        X: 初始迭代点
        eta: 收敛条件
        i_max: 最大迭代次数
    Returns: 方程的解X
    """
    ilucg_start = time.clock()
    i = 0 # 迭代次数
    R0 = np.dot(A, X)-B
    R_norm = np.linalg.norm(R0)
    # 求解My=r
    sA = sparse.csc_matrix(A)
    # 矩阵A的ILU分解
    lu = linalg.spilu(sA) 
    Y = lu.solve(R0)
    P = -Y
    
    while R_norm > eta and i < i_max:
        alpha = np.trace(np.dot(R0.T,Y)) / np.trace(np.dot(P.T, np.dot(A, P)))
        X = X + alpha * P
        R1 = R0 + alpha * np.dot(A, P)
        Y1 = lu.solve(R1)
        beta = np.trace(np.dot(R1.T, Y1)) / np.trace(np.dot(R0.T, Y))
        P = -Y1 + beta * P
        R0 = R1
        Y = Y1
        R_norm = np.linalg.norm(R0)
        i = i + 1
    ilucg_end = time.clock()
    
    return X,i,(ilucg_end-ilucg_start)*1000

def semiPosiM(size):
    A = np.random.rand(size, size) # 产生一个随机整数矩阵
    B = np.dot(A, A.T)

    return B  

if __name__=="__main__":
    size = 10
    A = semiPosiM(size)
    L = np.linalg.cholesky(A)
    R = np.random.rand(size, size)
    B = np.dot(A, R) # 产生一个随机整数矩阵
    X = np.zeros((size, size))
    eta = 1e-8
    i_max = 5000
    Xp,ip,ilucg_time = ilucg(A, B, X, eta, i_max)
    rss_pcg =np.square(np.linalg.norm(Xp-R))
    print(rss_pcg)
    print(ip)