# 导入核心库
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据生成函数
def generate_heterogeneous_data(seed=42):
    np.random.seed(seed)  # 固定随机种子，结果可复现
    # 簇A：200点，强正相关
    mu1 = np.array([1, 1])
    cov1 = np.array([[1.0, 0.8], [0.8, 1.0]])
    data_A = np.random.multivariate_normal(mu1, cov1, 200)
    # 簇B：400点，负相关
    mu2 = np.array([4, 4])
    cov2 = np.array([[1.2, -0.6], [-0.6, 0.8]])
    data_B = np.random.multivariate_normal(mu2, cov2, 400)
    # 合并数据与真实标签（0=簇A，1=簇B）
    data_all = np.vstack((data_A, data_B))
    label_all = np.hstack((np.zeros(200), np.ones(400)))
    return data_A, data_B, data_all, label_all

# 2. 生成数据并保存（作业要求：保留数据集）
data_A, data_B, data_all, label_all = generate_heterogeneous_data()
np.savez('gmm_data.npz', data_A=data_A, data_B=data_B, data_all=data_all, label_all=label_all)

# 3. 数据可视化
plt.figure(figsize=(8, 6))
plt.scatter(data_A[:, 0], data_A[:, 1], c='skyblue', label='Cluster A (200pts)', alpha=0.7)
plt.scatter(data_B[:, 0], data_B[:, 1], c='orange', label='Cluster B (400pts)', alpha=0.7)
plt.xlabel('X1 Dimension')
plt.ylabel('X2 Dimension')
plt.title('Heterogeneous Gaussian Distribution Data')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.savefig('data_generation.png', dpi=300)  # 保存可视化图
plt.show()

# 打印数据基本信息
print(f"簇A数据形状：{data_A.shape}，簇B数据形状：{data_B.shape}，合并数据形状：{data_all.shape}")