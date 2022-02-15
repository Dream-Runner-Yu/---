# 引入依赖
import numpy
import numpy as np
import pandas as pd

# 数据准备
# 评分矩阵

R = np.array([
    [4,0,2,0,1],
    [1,2,3,0,0],
    [1,0,2,4,0],
    [5,0,0,3,1],
    [0,0,1,5,1],
    [0,3,2,4,1],
])

print(R.shape)

K = 4
max_iter = 10000
alpha = 0.0002
lamba = 0.0001

def LFM_grad_desc( R, K=2, max_iter=1000, alpha=0.0001,lamba=0.002 ):
    # 基本维度
    M = len(R)
    N = len(R[0])

    P = np.random.rand(M,K)
    Q = np.random.rand(N,K)
    Q = Q.T

    for step in range(max_iter):
        #
        for u in range(M):
            for i in range(N):
                if R[u][i] > 0:
                    #
                    eui = np.dot(P[u,:],Q[:,i]) - R[u][i]

                    # 梯度下降更新 P Q
                    for k in range(K):
                        P[u][k] = P[u][k] - alpha * (2 * eui * Q[k][i] + 2 * lamba * P[u][k] )
                        Q[k][i] = Q[k][i] - alpha * (2 * eui * P[u][k] + 2 * lamba * Q[k][i])

        predR = np.dot(P,Q)

        cost = 0
        for u in range(M):
            for i in range(N):
                if R[u][i] > 0:
                    cost += (np.dot(P[u,:],Q[:,i])) ** 2
                    for k in range(K):
                        cost += lamba * (P[u][k] ** 2 + Q[k][i])
        if cost < 0.00001:
            break

    return P, Q.T, cost

P, Q, Cost  = LFM_grad_desc(R, K, max_iter, alpha,lamba)

predR = P.dot(Q.T)
print(P)
print(Q)
print(predR)
print(Cost)

