import numpy as np
import time
from scipy import sparse
from scipy.sparse import linalg

if __name__=="__main__":
    A = np.array([
        [1,2,0,4],
        [1,0,0,1],
        [1,0,2,1],
        [2,2,1,0.]])

    # X = np.array([
    #     [1,0,0,3],
    #     [-1,2,0,4],
    #     [2,3,4,1],
    #     [1,-1,0,1]])    
    #print(np.dot(A,X))
    B = np.array([
        [3,0,0,15],
        [2,-1,0,4],
        [6,5,8,6],
        [2,7,4,15]])
    # 必须要求是CSC格式            
    sA = sparse.csc_matrix(A)
    sb = sparse.csc_matrix(B)
    lu = linalg.spilu(sA)
    # b = np.array([1, 2, 3, 4])
    x = lu.solve(B)
    A.dot(x)
    print(A.dot(x))
    # L = lu.L.A
    # U = lu.U.A
    # print(np.dot(L,U))