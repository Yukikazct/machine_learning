import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams

warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'sans-serif']
rcParams['axes.unicode_minus'] = False
# 1. 设置均值（4种模式共用同一均值）
mean = [0, 0]
# 2. 定义4种协方差矩阵
cov_list = [
    [[1, 0], [0, 1]],       # 1. 各向同性（方差相同、无相关）
    [[2, 0], [0, 1]],       # 2. 非相关各向异性（对角、方差不同）
    [[1, 0.8], [0.8, 1]],   # 3. 相关各向异性（方差相同、有相关）
    [[2, 0.8], [0.8, 1]]    # 4. 相关各向异性（方差不同、有相关）
]
title_list = ["各向同性", "非相关各向异性", "相关各向异性(同方差)", "相关各向异性(异方差)"]

# 3. 生成数据并绘图
plt.figure(figsize=(16, 4))
for i in range(4):
    # 生成200个二维高斯点
    data = np.random.multivariate_normal(mean, cov_list[i], 200)
    plt.subplot(1, 4, i+1)
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6)
    plt.title(title_list[i])
    plt.axis("equal")  # 等比例坐标轴，看清形状
plt.show()







