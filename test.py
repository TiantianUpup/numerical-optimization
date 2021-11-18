import numpy as np

if __name__=="__main__":
    G  = np.array([[1, 2], [3, 4]])
    #G = np.matrix(g)
    a = np.array([1,1])
    b = np.array([2,3])
    # #print(b.shape)
    # print(np.dot(np.dot(b,G),b))
    print(np.dot(b,a))
    # a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # b = np.array([1,2,3])
    # # print(a.shape) #(3,3)
    # # print(b.shape) #(3,)
    # print(np.dot(a, b))