import numpy as np


class KMeans():

    def __init__(self, n_cluster, max_iter=100, e=0.0001):

        self.n_cluster = n_cluster  #K
        self.max_iter = max_iter    #迭代次数
        self.e = e

    def fit(self, x):

        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        K = self.n_cluster

        point_idx = np.arange(N)
        np.random.shuffle(point_idx)     # 随机打乱
        cluster_centers = point_idx[:K]  # K x 1
        mu = x[cluster_centers, :]       # 随机选取K个centers

        J = np.inf                       # 初始化，无穷大

        for i in range(self.max_iter):

            r = np.zeros(N)
            dist = np.zeros((N, K))

            for n in range(N):
                for k in range(K):
                    dist[n, k] = np.inner(mu[k,:]-x[n,:], mu[k,:]-x[n,:]) # 内积，每个像素点与K个center的欧氏距离
                
            r = np.argmin(dist, axis=1)  # N个像素点对应的最近K位置

            J_new = 0
            for n in range(N):
                J_new += dist[n,r[n]]   # n点与最近K的欧氏距离

            J_new /= N                  # 均值

            #print("Iteration [",i,"]: J = ", J ," ; Diff = ", np.absolute(J - J_new))
            print("Iteration [",i,"]: J = ", J)

            if np.absolute(J - J_new) <= self.e:
                return (mu, r, i)   # 返回K，每个像素点最近K，迭代次数
            
            J = J_new

            for k in range(K):
                k_idx_samples, = np.where(r == k)
                mu[k] = np.sum(x[k_idx_samples, :], axis=0) / len(k_idx_samples)# 更新质心K
 
        print("Did not converge!")
        return (mu, r, self.max_iter)
