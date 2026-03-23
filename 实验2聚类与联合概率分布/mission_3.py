import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from matplotlib import rcParams

warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'sans-serif']
rcParams['axes.unicode_minus'] = False

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

# ---------------------- 1. 读取并处理图片 ----------------------
img = Image.open("/Users/a/Desktop/机器学习/实验02/实验02-菜市场.jpg")
img = np.array(img)
H, W, C = img.shape
# 转换为(N,3)像素矩阵
pixels = img.reshape(-1, 3)

# ---------------------- 2. K-Means色彩聚类（K=7） ----------------------
labels, centers = kmeans(pixels, K=7)

# ---------------------- 3. 重构量化后图像 ----------------------
new_pixels = centers[labels].astype(np.uint8)
new_img = new_pixels.reshape(H, W, C)

# ---------------------- 4. 可视化结果（原图+量化图） ----------------------
plt.figure(figsize=(12,5))
plt.subplot(121); plt.imshow(img); plt.title("原图"); plt.axis("off")
plt.subplot(122); plt.imshow(new_img); plt.title("色彩量化(K=7)"); plt.axis("off")
plt.show()

# ---------------------- 5. 3D像素聚类可视化（核心修改：采样减少点数） ----------------------
from mpl_toolkits.mplot3d import Axes3D

# 方法1：固定采样数量（推荐，比如只取5000个点）
sample_size = 5000  # 可调整：越小点越少，建议5000-20000之间
# 生成随机采样索引（保证pixels和labels对应）
np.random.seed(0)  # 固定随机种子，结果可复现
sample_idx = np.random.choice(len(pixels), size=sample_size, replace=False)
pixels_sample = pixels[sample_idx]
labels_sample = labels[sample_idx]

# 方法2：按比例采样（比如取10%的点）
# sample_ratio = 0.1
# sample_idx = np.random.choice(len(pixels), size=int(len(pixels)*sample_ratio), replace=False)
# pixels_sample = pixels[sample_idx]
# labels_sample = labels[sample_idx]

# 绘制3D图
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
# 只绘制采样后的点
ax.scatter(pixels_sample[:,0], pixels_sample[:,1], pixels_sample[:,2],
           c=labels_sample, s=1, alpha=0.5)
# 聚类中心仍全部绘制（红色星号）
ax.scatter(centers[:,0], centers[:,1], centers[:,2], c="red", s=100, marker="*")
ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
ax.set_title("RGB像素聚类结果(K=7)（采样5000个点）")
plt.show()