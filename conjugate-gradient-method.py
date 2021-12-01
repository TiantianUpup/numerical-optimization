import numpy as np

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
    r = b - np.dot(A, x)
    rList = []
    rList.append(r)
    r_norm = np.linalg.norm(r, ord=None, axis=None, keepdims=False)
    while r_norm >= eta and i < i_max:
        i = i + 1
        if (i == 1):
            p = rList[0]
        else:
            # \进行换行
            beta = np.square(np.linalg.norm(rList[i-1], ord=None, axis=None, keepdims=False)) /\
                 np.square(np.linalg.norm(rList[i-2], ord=None, axis=None, keepdims=False))
            p = rList[i-1] + beta * p
        alpha = np.square(np.linalg.norm(rList[i-1], ord=None, axis=None, keepdims=False)) / np.dot(p, np.dot(A, p))
        r = r - alpha * np.dot(A, p)
        rList.append(r)
        r_norm = np.linalg.norm(r, ord=None, axis=None, keepdims=False)
        x = x + alpha * p

    return x    

if __name__=="__main__":
    A = np.array([
        [1,0,0],
        [0,2,0],
        [0,0,4]
    ])
    b = np.array([0,4,2])
    x = np.array([0,0,0])
    eta = 1e-8
    i_max = 100
    x = cg(A, b, x, eta, i_max)
    print("resullt is:{x}".format(x=x))