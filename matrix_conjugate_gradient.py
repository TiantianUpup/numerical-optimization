import numpy as np
import time
import math

def cg(A, B, X, eta, i_max):
    """共轭梯度法求解方程AX=B，这里A为对称正定矩阵
    Args:
        A: 方程系数矩阵
        B:
        X: 初始迭代点
        eta: 收敛条件
        i_max: 最大迭代次数
    Returns: 方程的解X
    """
    cg_start = time.clock()
    i = 0 # 迭代次数
    R0 = np.dot(A, X)-B
    P = -R0
    R_norm = np.linalg.norm(R0)
    while R_norm > eta and i < i_max:
        alpha = np.trace(np.dot(R0.T,R0)) / np.trace(np.dot(P.T, np.dot(A, P)))
        X = X + alpha * P
        R1 = R0 + alpha * np.dot(A, P)
        
        beta = np.trace(np.dot(R1.T, R1)) / np.trace(np.dot(R0.T, R0))
        P = -R1 + beta * P
        R0 = R1
        R_norm = np.linalg.norm(R0)
        i = i + 1
    cg_end = time.clock()
    
    return X,i,(cg_end-cg_start)*1000

if __name__=="__main__":
    A = np.array([
        [1, 1, 0],
        [1, 4, 1],
        [0, 1, 10]])
    
    B = np.array([
        [1, 3, 1],
        [1, 9, 7],
        [0, 2, 31]])

    X = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ]) 
    
    result = np.array([
        [1, 1, 0],
        [0, 2, 1],
        [0, 0, 3]])
    eta = 1e-8
    i_max = 100
    X = cg(A, B, X, eta, i_max)
    rss_cg =np.square(np.linalg.norm(X-result)**2)
    print(rss_cg)
    print(X)