import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===================== 全局配置：固定随机种子，保证结果可复现 =====================
np.random.seed(42)

# ===================== 步骤1：参数配置（可灵活修改） =====================
SAMPLE_NUM = 100  # 样本量
# 特征取值范围
ALT_RANGE = (0, 2000)  # 海拔范围
LAT_RANGE = (0, 90)    # 纬度范围
# 真实权重（实验指定）
W_TRUE = np.array([30.0, -0.006, -0.2])
NOISE_MEAN, NOISE_STD = 0, 1  # 高斯噪声参数
DATASET_PATH = "temperature_dataset.npz"


def generate_and_save_dataset():
    # 生成原始特征
    x1 = np.random.uniform(*ALT_RANGE, SAMPLE_NUM)
    x2 = np.random.uniform(*LAT_RANGE, SAMPLE_NUM)
    # 生成带噪声的标签
    X_raw = np.c_[np.ones(SAMPLE_NUM), x1, x2]
    epsilon = np.random.normal(NOISE_MEAN, NOISE_STD, SAMPLE_NUM)
    y = X_raw @ W_TRUE + epsilon

    np.savez_compressed(
        DATASET_PATH,
        x1=x1,    # 海拔
        x2=x2,    # 纬度
        y=y       # 温度标签
    )
    return x1, x2, y


# ===================== 步骤3：加载已保存的 NPZ 数据集 =====================
def load_saved_dataset():
    """从 NPZ 文件加载已保存的数据集（纯NumPy读取）"""
    print("===== 正在加载本地数据集 =====")
    # 加载NPZ文件（类字典对象）
    data = np.load(DATASET_PATH)
    # 提取数组
    x1 = data["x1"]
    x2 = data["x2"]
    y = data["y"]
    print(f"✅ 成功加载数据集，样本量：{len(x1)}")
    return x1, x2, y


# ===================== 主程序：选择 生成新数据 / 加载本地数据 =====================
# 首次运行：生成并保存数据 → 后续运行：注释下一行，取消注释加载行
x1, x2, y = generate_and_save_dataset()
# x1, x2, y = load_saved_dataset()

# 构建设计矩阵 X ∈ R^100×3（第一列全1=偏置项，第二列x1，第三列x2）
X = np.c_[np.ones(SAMPLE_NUM), x1, x2]
print(f"\n【步骤A】设计矩阵X形状: {X.shape}（100行3列，符合要求）")

# ===================== 步骤C：满秩性验证 =====================
rank_X = np.linalg.matrix_rank(X)
D = X.shape[1]
is_full_rank = rank_X == D
print(f"【步骤C】设计矩阵X的秩: {rank_X}，特征维度D: {D}，是否满秩: {is_full_rank}")

# ===================== 步骤D：正规方程求解与评估 =====================
X_T = X.T
X_T_X = X_T @ X
w_hat = np.linalg.inv(X_T_X) @ X_T @ y  # 最优权重
eu_dist = np.linalg.norm(w_hat - W_TRUE)  # 欧式距离
print(f"【步骤D】求解的最优权重w_hat: {np.round(w_hat, 5)}")
print(f"【步骤D】w_hat与w_true的欧氏距离: {np.round(eu_dist, 5)}")

# ===================== 3D可视化：原始数据+预测超平面 =====================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, c='blue', marker='o', s=50, label='Raw Data (with Noise)')
# 生成超平面网格
x1_mesh, x2_mesh = np.meshgrid(np.linspace(*ALT_RANGE, 20), np.linspace(*LAT_RANGE, 20))
y_mesh = w_hat[0] + w_hat[1] * x1_mesh + w_hat[2] * x2_mesh
ax.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.5, color='red', label='Prediction Hyperplane')
# 图表设置
ax.set_xlabel('Altitude x1', fontsize=12)
ax.set_ylabel('Latitude x2', fontsize=12)
ax.set_zlabel('Temperature y', fontsize=12)
ax.set_title('3D Prediction Hyperplane of Linear Regression', fontsize=14, pad=20)
ax.legend(loc='upper right')
plt.savefig('Experiment1_Temperature_Prediction_Hyperplane.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 步骤E：几何解释 - 正交投影验证 =====================
y_hat = X @ w_hat
r = y - y_hat
dot_col0 = np.dot(r, X[:, 0])
dot_col1 = np.dot(r, X[:, 1])
dot_col2 = np.dot(r, X[:, 2])
print(f"\n【步骤E】残差与X第1列（偏置全1）的点积: {np.round(dot_col0, 5)}")
print(f"【步骤E】残差与X第2列（海拔x1）的点积: {np.round(dot_col1, 5)}")
print(f"【步骤E】残差与X第3列（纬度x2）的点积: {np.round(dot_col2, 5)}")

print("\n【实验1完成】所有步骤执行完毕！")