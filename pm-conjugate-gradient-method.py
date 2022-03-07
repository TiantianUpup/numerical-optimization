import numpy as np

def pcg(A, B, X, M, eta, i_max):
    """共轭梯度法求解矩阵方程AX=B，这里A为对称正定矩阵
    Args:
        A: 方程系数矩阵，为正定矩阵
        B:
        X: 初始迭代点，一般选取为0
        M: 预处理矩阵
        eta: 收敛条件
        i_max: 最大迭代次数
    Returns: 方程的解X
    """
    i = 0 # 迭代次数
    R = B - np.dot(A, X)
    rList = []
    rList.append(R)
    r_norm = np.linalg.norm(R) # 默认计算的是矩阵的F-范数
    while r_norm >= eta and i < i_max:
        i = i + 1
        if (i == 1):
            p = rList[0]
        else:
            beta = np.square(np.linalg.norm(rList[i-1])) / np.square(np.linalg.norm(rList[i-2]))
            p = rList[i-1] + beta * p
        alpha = np.square(np.linalg.norm(rList[i-1])) /np.trace( np.dot(p.T, np.dot(A, p)))
        
        R = R - alpha * np.dot(A, p)
        rList.append(R)
        r_norm = np.linalg.norm(R)
        X = X + alpha * p

    return X

if __name__=="__main__":
    A = np.array([
        [1,2],
        [2,5]
    ])
    B = np.array([
        [5,4],
        [12,9]
    ])
    X = np.array([
        [0,0],
        [0,0]
    ]) 
    C = np.array([
        [1,1],
        [1,1]
    ]) 
    eta = 1e-8
    i_max = 100
    x = pcg(A, B, X, eta, i_max)
    print("resullt is:{x}".format(x=x))
    #R = B - np.dot(A, X)
    #print(np.linalg.norm(R))
    #print(np.dot(A,C)) s