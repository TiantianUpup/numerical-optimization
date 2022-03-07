import numpy as np

def semiPosiM(size):
    A = -1+2*np.random.rand(size,size) # 产生一个随机整数矩阵
    B = np.dot(A, A.T)
    return B   

if __name__=="__main__":
    # size = 5
    A = semiPosiM(-1,2,5)
    # A = -1+2*np.random.rand(5,5)

    # print("A is \n {A}".format(A=A))
    # B = matrix1 = np.random.uniform(-2, 3, (5, 5)) # 生成一个随机[-2,3)的 5*5的随机矩阵
    # print("B is \n {B}".format(B=B))
    # print(A.dot(B))
    #print(np.zeros((5,5)))
    # print("A is:\n{A}".format(A=A))
    # print("=======================================================")
    # L = np.linalg.cholesky(A)
    # print("L is:\n{L}".format(L=L))
    # print("=======================================================")
    # print(L.dot(L.T))
    # print(A.shape) 
    # print(np.trace(A))
    print(A)