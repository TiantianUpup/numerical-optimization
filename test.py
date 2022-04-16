import numpy as np

def projectToSpectraplex(X):
    """Spectraplex上的投影
    Args:
        X: 对称矩阵
    Returns: 矩阵X在Spectraplex上的投影   
    """
    e_vals, U = np.linalg.eig(X)
    e_vals, U = quickSortEVD(e_vals,U)
    return np.dot(U, np.dot(np.diag(projectToSimplex(e_vals)), U.T))

def projectToSimplex(x):
    """单纯形上的投影
    Args:
        x: 数组[向量]
    Return:
        x往单纯形上的投影    
    """
    size = x.size
    e = np.ones(size)
    k = (1-np.dot(e,x))/size
    u = x + k*e

    if isNonnegativeVector(u):
        return u
    else:
        z = quickSort(u.copy())
        s_1 = 0
        s_2 = z[0]
        gamma = 0
        for i in range(1, z.size-1):
            s_1 = s_1 + z[i-1]
            s_2 = s_2 + z[i]
            if s_1 - i * z[i-1] < 1 and s_2 - (i + 1) * z[i] > 1:
                gamma = (s_1 - 1) / i
                break
            if s_1 - i * z[i-1] < 1 and s_2 - (i + 1) * z[i] == 1:
                gamma = z[i]
                break
       
        return projectionToNonnegativeOrthant(projectionToNonnegativeOrthant(u)-gamma * e) 

def quickSort(y, left=None, right=None):
    """快排，从大到小进行排序

    """
    left = 0 if not isinstance(left,(int, float)) else left
    right = len(y)-1 if not isinstance(right,(int, float)) else right
    if left < right:
        partitionIndex = partition(y, left, right)
        quickSort(y, left, partitionIndex-1)
        quickSort(y, partitionIndex+1, right)
    return y

def partition(y, left, right):
    pivot = left
    index = pivot+1
    i = index
    while  i <= right:
        if y[i] > y[pivot]:
            swap(y, i, index)
            index+=1
        i+=1
    swap(y,pivot,index-1)
    return index-1

def swap(y, i, j):
    y[i], y[j] = y[j], y[i]

def quickSortEVD(y, X,left=None, right=None):
    """快排，从大到小进行排序

    """
    left = 0 if not isinstance(left,(int, float)) else left
    right = len(y)-1 if not isinstance(right,(int, float)) else right
    if left < right:
        partitionIndex = partitionEVD(y, X, left, right)
        quickSortEVD(y, X, left, partitionIndex-1)
        quickSortEVD(y, X, partitionIndex+1, right)
    return y,X

def partitionEVD(y, X, left, right):
    pivot = left
    index = pivot+1
    i = index
    while  i <= right:
        if y[i] > y[pivot]:
            swapEVD(y, X, i, index)
            index+=1
        i+=1
    swapEVD(y, X,pivot,index-1)
    return index-1

def swapEVD(y, X,i, j):
    X[:,i], X[:,j] = np.copy(X[:,j]), np.copy(X[:,i])
    y[i], y[j] = y[j], y[i]

def isNonnegativeVector(x):
    """判断向量是否为非负向量
    Args:
        x:数组[向量]
    Return:
        是非负数组返回True，否则返回False    
    """
    for i in range(x.size):
        if x[i] < 0:
            return False
    return True;    

def projectionToNonnegativeOrthant(x):
    """计算向量到非负卦限上的投影
    Args:
        x: 投影的向量
    Returns: x在非负卦限上的投影    
    """
    for i in range(x.size):
        if x[i] < 0:
            x[i] = 0 
    
    return x         

if __name__=="__main__":
    x = np.array([5,3,2,1])
    
    #print(projectToSimplex(x))
    # U = np.array([
    #     [1,2,3],
    #     [1,2,3],
    #     [1,2,3]
    # ])
    # print("before exchange:{U}".format(U=U))
    # print(U[:,1])
    # print(U.shape)
    # A = np.full(U.shape, 0)
    # A[:,0] = U[:,1]
    # # A[:,[1]] = U[:,0]
    # U[:,0], U[:,1] = np.copy(U[:,1]), np.copy(U[:,0])
    # print("after exchange:{U}".format(U=U))
    X = np.array([
        [1,4,2],
        [4,2,2],
        [2,2,1]
    ])
    # X = np.array([
    #     [3,0,0,0],
    #     [0,5,0,0],
    #     [0,0,2,0],
    #     [0,0,0,1]
    # ])
    print(projectToSpectraplex(X))
