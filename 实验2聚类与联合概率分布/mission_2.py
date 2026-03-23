import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 生成实验数据 ----------------------
# 4个均值，共用协方差
means = [[0,0],[3,1],[1,4],[4,4]]
cov = [[1,0],[0,1]]
data = []
for m in means:
    # 每组200个点，共800个
    data.append(np.random.multivariate_normal(m, cov, 200))
data = np.vstack(data)  # 拼接成(800,2)

# ---------------------- 2. K-Means核心函数 ----------------------
def kmeans(data, K, max_iter=100, tol=1e-6):
    """
    data: 样本(N,D)
    K: 聚类数
    max_iter: 最大迭代次数
    tol: 收敛阈值（中心变化小于此值停止）
    """
    N, D = data.shape
    # 随机初始化中心（从数据中选K个点）
    centers = data[np.random.choice(N, K, replace=False)]
    for _ in range(max_iter):
        # 步骤1：分配样本（计算距离→找最近中心）
        dists = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(dists, axis=0)
        # 步骤2：更新中心
        new_centers = np.array([data[labels==k].mean(axis=0) for k in range(K)])
        # 收敛判断
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    return labels, centers

# ---------------------- 3. 实验对比 ----------------------
# 任务A：K=4
labels4, centers4 = kmeans(data, K=4)
# 任务B：K=8
labels8, centers8 = kmeans(data, K=8)

# 绘图对比
plt.figure(figsize=(12,5))
plt.subplot(121); plt.scatter(data[:,0], data[:,1], c=labels4, s=10); plt.title("K=4")
plt.scatter(centers4[:,0], centers4[:,1], c="red", s=100, marker="*")
plt.subplot(122); plt.scatter(data[:,0], data[:,1], c=labels8, s=10); plt.title("K=8")
plt.scatter(centers8[:,0], centers8[:,1], c="red", s=100, marker="*")
plt.show()