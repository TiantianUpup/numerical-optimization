from telnetlib import XASCII
import numpy as np
import pre_conjugate_gradient as pcg
import matrix_conjugate_gradient as mcg

def semiPosiM(size):
    A = np.random.rand(size, size) # 产生一个随机整数矩阵
    B = np.dot(A, A.T)
    return B    

if __name__=="__main__":
    size = 500
    # 循环测试五次
    A = semiPosiM(size)
    L = np.linalg.cholesky(A)
    R = np.random.rand(size, size)
    B = np.dot(A, R) # 产生一个随机整数矩阵
    X = np.zeros((size, size))
    eta = 1e-8
    i_max = 5000

    Xp,ip,ilucg_time = pcg.ilucg(A, B, X, eta, i_max)
    rss_pcg =np.linalg.norm(Xp-R)**2
    print("pcg iten is {i}, ilucg_time is {t} ms, rss_pcg is {rss}".format(i=ip,t = ilucg_time, rss = rss_pcg))
    X,i,cg_time = mcg.cg(A, B, X, eta, i_max)
    rss_cg = np.linalg.norm(X-R)**2
    print("cg iten is {i}, ilucg_time is {t} ms, rss_cg is {rss}".format(i=i,t = cg_time, rss = rss_cg))
    # for i in range(5):
    #     Xp,ip,ilucg_time = pcg.ilucg(A, B, X, eta, i_max)
    #     rss_pcg =np.square(np.linalg.norm(Xp-R)**2)
    #     X,ip,ilucg_time = mcg.cg(A, B, X, eta, i_max)
    #     rss_cg =np.square(np.linalg.norm(X-R)**2)
    
   
