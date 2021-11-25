import numpy as np

if __name__=="__main__":
    G  = np.array([[21, 4], [4, 15]])
    # G  = np.array([[21, 4], [4, 1]])
    b = np.array([2,3])
    x = np.array([-30,100]) # 初始点
    g = -(np.dot(G,x)+b) # 梯度
    i = 0
    g_norm=np.linalg.norm(g, ord=None, axis=None, keepdims=False)
    # 1. 判断是否满足终止条件
    while g_norm >= 1e-6:
        i = i + 1
        print("iterate {i} times".format(i=i))
        # 2.精确求步长
        alpha = - (np.dot(x,np.dot(G,g))+np.dot(b,g)) / (np.dot(g,np.dot(G,g)))
        # 3.求下一个迭代点
        x = x + alpha * g
        print(x)
        # 4.计算梯度及梯度范数
        g = -(np.dot(G,x)+b)
        g_norm=np.linalg.norm(g, ord=None, axis=None, keepdims=False)
        print(g_norm)

