import numpy as np
import time
from scipy import sparse
from scipy.sparse import linalg

def cg(A, B, X, eta, i_max):
    cg_start = time.clock()
    """共轭梯度法求解矩阵方程AX=B，这里A为对称正定矩阵
    Args:
        A: 方程系数矩阵，为正定矩阵
        B:
        X: 初始迭代点，一般选取为0
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
        #print("cg method...")
        i = i + 1
        if (i == 1):
            p = rList[0]
        else:
            beta = np.square(np.linalg.norm(rList[i-1])) / np.square(np.linalg.norm(rList[i-2]))
            p = rList[i-1] + beta * p
        alpha = np.square(np.linalg.norm(rList[i-1])) /np.trace(np.dot(p.T, np.dot(A, p)))
        # print("p is:{p}\n".format(p=p))
        # print("matrix is:\n{matrix}".format(matrix = np.dot(p.T, np.dot(A, p))))
        # print("trace is:{trace}".format(trace = np.trace(np.dot(p.T, np.dot(A, p)))))
        # print("================================ Ending ===============================")
        R = R - alpha * np.dot(A, p)
        rList.append(R)
        r_norm = np.linalg.norm(R)
        X = X + alpha * p

    cg_end = time.clock()
    return X,i,(cg_end - cg_start)*1000

def ilucg(A, B, X, eta, i_max):
    """预处理共轭梯度法求解矩阵方程AX=B，这里A为对称正定矩阵
    Args:
        A: 方程系数矩阵，为正定矩阵
        B:
        X: 初始迭代点，一般选取为0
        eta: 收敛条件
        i_max: 最大迭代次数
    Returns: 方程的解X
    """
    ilucg_start = time.clock()
    i = 0 # 迭代次数
    R = np.dot(A, X) - B #R_0
    # print(R)
    sA = sparse.csc_matrix(A)
    # 矩阵A的ILU分解
    lu = linalg.spilu(sA) 
    Y = lu.solve(R)
    # print(Y)
    # print(A.dot(Y))
    P = -Y
    r_norm = np.linalg.norm(R) # 默认计算的是矩阵的F-范数
    # print(R.T)
    # print(Y)
    # print(np.dot(R.T, Y))
    while r_norm >= eta and i < i_max:
       #print("ilucg method...")
    #    print("R.TY trace is:{trace}".format(trace=np.trace(np.dot(R.T, Y))))
    #    print("P.TAP trace is:{trace}".format(trace=np.trace(np.dot(P.T, np.dot(A, P)))))
       alpha = np.trace(np.dot(R.T, Y)) / np.trace(np.dot(P.T, np.dot(A, P)))
       #print(alpha)
       X = X + alpha * P
       R = R + alpha * np.dot(A, P)
       #print("matrix R is==============:\{R}".format(R=R))
       # 求解方程My=r
       Y = lu.solve(R)
       #print("matrix Y is==============:\{Y}".format(Y=Y))
       beta = np.trace(np.dot(R.T, Y))
       P = -Y + beta * P
       #print("matrix P is==============:\{P}".format(P=P))
       i = i + 1
       r_norm = np.linalg.norm(R)

    ilucg_end = time.clock() 

    return X,i,(ilucg_end - ilucg_start) * 1000

def semiPosiM(size):
    A = -1+2*np.random.rand(size,size) # 产生一个随机整数矩阵
    return np.dot(A, A.T) 

if __name__=="__main__":
    # A = np.array([
    #     [1,2],
    #     [2,5]
    # ])
    # B = np.array([
    #     [5,4],
    #     [12,9]
    # ])
    # X = np.array([
    #     [0,0],
    #     [0,0]
    # ]) 
    # C = np.array([
    #     [1,1],
    #     [1,1]
    # ]) 

    size = 10
    A =  semiPosiM(size) # 共轭梯度法必须保证矩阵为正定
    #A = np.random.randint(-1,2,(10,10))
    #print("A is ============================ \n {A}".format(A=A))
    B =np.random.rand(size,size) # 生成一个随机[0,1]的矩阵
    #print("B is ============================ \n {B}".format(B=B))
    AB = A.dot(B)
    #print("AB is =========================== \n {AB}".format(AB=AB))
    X = np.zeros((size,size))
    eta = 1e-8
    i_max = 5000
    

    x,i,ilucg_time = ilucg(A, AB, X, eta, i_max)
    rss_ilucg =np.square(np.linalg.norm(x-B))
    y,j,cg_time = cg(A, AB, X, eta, i_max)
    rss_cg =np.square(np.linalg.norm(y-B))
    print("====================== Algorithm Ending ============================")
    print("ilucg iterate:{i}, time cost is:{time}, rss is:{rss}, the solution is".format(i=i,time = ilucg_time,rss=rss_ilucg,x=x))
    # print("ilucg cost is:{time}".format(time = ilucg_time))
    #print("====================== Result Spillt ===============================")
    print("cg iterate is:{j}, time cost is:{time}, rss is:{rss}, the solution is".format(j=j,time=cg_time,rss=rss_cg,y=y))
    #print("cg cost is:{time}".format(time =cg_time))

