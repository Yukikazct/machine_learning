import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

# 步骤A：构建合成数据集
N = 100  # 样本量
x1 = np.random.uniform(0, 2000, N)  # 海拔：[0,2000]均匀分布
x2 = np.random.uniform(0, 90, N)    # 纬度：[0,90]均匀分布
# 构建设计矩阵X∈R^100×3：第一列全1（偏置项），第二列x1，第三列x2
X = np.c_[np.ones(N), x1, x2]
print(f"【步骤A】设计矩阵X形状: {X.shape}（100行3列，符合要求）")

#步骤B：生成观测标签y
w_true = np.array([30.0, -0.006, -0.2])  # 实验手册指定
epsilon = np.random.normal(0, 1, N)      # 高斯噪声ε~N(0,1)
y = X @ w_true + epsilon                 # 计算带噪声标签：y=Xw_true+ε
print(f"【步骤B】标签y形状: {y.shape}，真实权重w_true: {w_true}")

#步骤C：满秩性验证
rank_X = np.linalg.matrix_rank(X)  # 计算设计矩阵X的秩
D = X.shape[1]                     # 特征维度（含偏置，D=3）
is_full_rank = rank_X == D
print(f"【步骤C】设计矩阵X的秩: {rank_X}，特征维度D: {D}，是否满秩: {is_full_rank}")

#步骤D：正规方程求解与评估
# 计算正规方程各部分矩阵
X_T = X.T
X_T_X = X_T @ X
# 正规方程核心：w_hat = (X^T X)^{-1} X^T y
w_hat = np.linalg.inv(X_T_X) @ X_T @ y
# 计算w_hat与w_true的欧几里得距离
eu_dist = np.linalg.norm(w_hat - w_true)
# 打印结果
print(f"【步骤D】求解的最优权重w_hat: {np.round(w_hat, 5)}")
print(f"【步骤D】w_hat与w_true的欧氏距离: {np.round(eu_dist, 5)}")

# 3D可视化：原始数据+预测超平面 (英文版本)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# 绘制带噪声的原始数据散点
ax.scatter(x1, x2, y, c='blue', marker='o', s=50, label='Raw Data (with Noise)')
# 生成网格点，用于绘制平滑的预测超平面
x1_mesh, x2_mesh = np.meshgrid(np.linspace(0, 2000, 20), np.linspace(0, 90, 20))
y_mesh = w_hat[0] + w_hat[1] * x1_mesh + w_hat[2] * x2_mesh
# 绘制线性回归预测超平面
ax.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.5, color='red', label='Prediction Hyperplane')
# 坐标轴与图例设置
ax.set_xlabel('Altitude x1', fontsize=12)
ax.set_ylabel('Latitude x2', fontsize=12)
ax.set_zlabel('Temperature y', fontsize=12)
ax.set_title('3D Prediction Hyperplane of Linear Regression (Altitude-Latitude-Temperature)', fontsize=14, pad=20)
ax.legend(loc='upper right')
# 保存图片
plt.savefig('Experiment1_Temperature_Prediction_Hyperplane.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 步骤E：几何解释 - 正交投影验证 =====================
y_hat = X @ w_hat  # 模型预测值向量
r = y - y_hat      # 残差向量：真实值-预测值
# 计算残差与设计矩阵X每一列的点积（验证正交性）
dot_col0 = np.dot(r, X[:, 0])  # 与偏置列（全1）的点积
dot_col1 = np.dot(r, X[:, 1])  # 与海拔列x1的点积
dot_col2 = np.dot(r, X[:, 2])  # 与纬度列x2的点积
# 打印点积结果
print(f"【步骤E】残差与X第1列（偏置全1）的点积: {np.round(dot_col0, 5)}")
print(f"【步骤E】残差与X第2列（海拔x1）的点积: {np.round(dot_col1, 5)}")
print(f"【步骤E】残差与X第3列（纬度x2）的点积: {np.round(dot_col2, 5)}")

print("\n【实验1完成】所有步骤执行完毕！")