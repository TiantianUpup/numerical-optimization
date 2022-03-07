import numpy as np

if __name__=="__main__":
    A = np.array([[1,2],
    [2,5]])
    try:
        L = np.linalg.cholesky(A)
        print('is execute')
    except np.linalg.LinAlgError:
        print("Matrix is not positive definite")    
