import numpy as np

def pg(A, b, x, eta, i_max):
    
    return 0

if __name__=="__main__":
    A = np.array([
        [2,0,0],
        [0,2,0],
        [0,0,4]
    ])
    e_vals,e_vecs = np.linalg.eig(A)
    print(e_vals)
    print(e_vecs)