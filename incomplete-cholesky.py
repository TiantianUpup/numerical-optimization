import numpy as np
import time

def icholesky(a):
    n = a.shape[0]
    for k in range(n): 
        if a[k,k] < 0:
            a[k,k] = 0
        else:    
            a[k,k] = np.sqrt(a[k,k])
        
        for i in range(k+1,n):
           if a[i,k] !=0:
               a[i,k] = a[i,k]/a[k,k]
        
        for j in range(k+1,n):
            for i in range(j,n):
                if a[i,j]!=0:
                    a[i,j] = a[i,j]-a[i,k]*a[j,k]         

    for i in range(n):
        for j in range(i+1, n):
            a[i,j] = 0     

    return a

def semiPosiM(low, high, size):
    A = np.random.randint(low, high, size=(size, size)) # 产生一个随机整数矩阵
    B = np.dot(A, A.T)
    return B

if __name__=="__main__":
    a = semiPosiM(-1,2,5)
    
    print(a)
    # ich_start = time.clock()
    L1=icholesky(a)
    # ich_end = time.clock()
    # print("ich cost is:{time}".format(time=ich_end-ich_start))
    # print(L1)
    # # print(np.dot(L1,L1.T))
    # #print(a)
    # ch_start = time.clock()
    # L2 = np.linalg.cholesky(b)
    # ch_end = time.clock()
    # print("ch cost is:{time}".format(time=ch_end-ch_start))
    # print(L2)
    # ich_start = time.clock()
    # L1=icholesky(a)
    # ich_end = time.clock()
    # print("ich cost is:{time}".format(time=ich_end-ich_start))
    # print(L1)
    #print(np.dot(L2,L2.T))