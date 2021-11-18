import numpy as np

if __name__=="__main__":
    # 无约束优化问题:min f(x) = 3x_1^2+3x_2^2-x_1^2x_2
    alpha = 1 # 牛顿法中步长固定为1
    x = np.array([1.5,1.5]) # 初始点
    g = np.array([6 * x[0] - 2*x[0]*x[1],6*x[1]-x[1]*x[1]])
    G = np.array([[6-2*x[1],-2*x[0]],[-2*x[0],6]])
    g_norm=np.linalg.norm(g, ord=None, axis=None, keepdims=False)
    i = 0
    while g_norm >= 1e-7:
        i = i + 1
        print("iterate {i} times".format(i=i))
        d = - np.dot(np.linalg.inv(G),g)
        # print(d)
        x = x + d
        print(x)
        g = np.array([6 * x[0] - 2*x[0]*x[1],6*x[1]-x[0]*x[0]])
        G = np.array([[6-2*x[1],-2*x[0]],[-2*x[0],6]])
        g_norm=np.linalg.norm(g, ord=None, axis=None, keepdims=False)
        print(g_norm)
        print("f_value is {value}".format(value=3*x[0]*x[0]+3*x[1]*x[1]-x[0]*x[0]*x[1]))
