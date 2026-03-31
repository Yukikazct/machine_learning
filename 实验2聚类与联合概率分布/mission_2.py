import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# 固定随机种子，保证实验结果可复现
np.random.seed(42)

# ---------------------- 1. 生成实验数据（题目要求） ----------------------
# 4个聚类中心均值，共用协方差矩阵
means = [[0, 0], [3, 1], [1, 4], [4, 4]]
cov = [[1, 0], [0, 1]]
data = []
# 生成真实标签，用于计算聚类准确率
true_labels = []
for i, m in enumerate(means):
    points = np.random.multivariate_normal(m, cov, 200)
    data.append(points)
    true_labels.extend([i] * 200)

data = np.vstack(data)
true_labels = np.array(true_labels)


# ---------------------- 2. 手动实现K-Means算法 ----------------------
def kmeans(data, K, max_iter=100, tol=1e-6):
    N, D = data.shape
    # 随机初始化聚类中心
    centers = data[np.random.choice(N, K, replace=False)]

    for _ in range(max_iter):
        # 计算欧氏距离，分配簇
        dists = np.sqrt(((data - centers[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(dists, axis=0)

        # 更新聚类中心
        new_centers = np.array([data[labels == k].mean(axis=0) for k in range(K)])

        # 收敛判断
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers

    return labels, centers


# ---------------------- 3. 计算聚类准确率（聚合准确度） ----------------------
def clustering_accuracy(y_true, y_pred):
    cost = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cost[i, j] = np.sum((y_true == i) & (y_pred == j))
    row_ind, col_ind = linear_sum_assignment(-cost)
    acc = cost[row_ind, col_ind].sum() / len(y_true)
    return acc


# ---------------------- 4. 执行聚类实验 ----------------------
labels4, centers4 = kmeans(data, K=4)
labels8, centers8 = kmeans(data, K=8)

# 计算聚合准确度
acc4 = clustering_accuracy(true_labels, labels4)

# 控制台输出中文结果（直接填写实验报告）
print("===== 实验结果 =====")
print(f"K=4 聚类中心：\n{centers4}")
print(f"K=4 聚类聚合准确度：{acc4 * 100:.2f}%")
print("K=8 出现过度聚类现象")

# ---------------------- 5. 可视化（图表纯英文，其余中文） ----------------------
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1], c=labels4, s=10)
plt.scatter(centers4[:, 0], centers4[:, 1], c="red", s=100, marker="*", label="Cluster Centers")
plt.title("K-Means Clustering Result (K=4)")
plt.legend()

# K=8 子图
plt.subplot(122)
plt.scatter(data[:, 0], data[:, 1], c=labels8, s=10)
plt.scatter(centers8[:, 0], centers8[:, 1], c="red", s=100, marker="*", label="Cluster Centers")
plt.title("K-Means Clustering Result (K=8)")
plt.legend()

plt.tight_layout()
plt.show()