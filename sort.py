import numpy as np
def quickSortEVD(X, y, left=None, right=None):
    """快排，从大到小进行排序

    """
    left = 0 if not isinstance(left,(int, float)) else left
    right = len(y)-1 if not isinstance(right,(int, float)) else right
    if left < right:
        partitionIndex = partitionEVD(X, y, left, right)
        quickSortEVD(X, y, left, partitionIndex-1)
        quickSortEVD(X, y, partitionIndex+1, right)
    return y,X

def partitionEVD(X, y, left, right):
    pivot = left
    index = pivot+1
    i = index
    while  i <= right:
        if y[i] > y[pivot]:
            swapEVD(X, y, i, index)
            index+=1
        i+=1
    swapEVD(X, y,pivot,index-1)
    return index-1

def swapEVD(X, y, i, j):
    X[:,i], X[:,j] = np.copy(X[:,j]), np.copy(X[:,i])
    y[i], y[j] = y[j], y[i]

if __name__=="__main__":
    X = np.array([
        [4,0,0],
        [0,1,0],
        [0,0,2]
    ])
    e_vals,e_vecs = np.linalg.eig(X)
    # print(e_vals)
    # print("======= half spilt =======")
    # print(e_vecs)
    # # print("======= spilt =======")
    # # y, U = quickSortD(e_vals, e_vecs)
    # # print(y)
    # # print("======= half spilt =======")
    # # print(U)
    # #
    y, U = quickSortEVD(e_vecs, e_vals)
    print(y)
    print("======= half spilt =======")
    print(U)
    # A = np.array([
    #     [1,2],
    #     [4,3]
    # ])
    # B = np.array([
    #     [1,4],
    #     [2,2]
    # ])
    # print(np.dot(A, np.dot(B,A.T)))
    
