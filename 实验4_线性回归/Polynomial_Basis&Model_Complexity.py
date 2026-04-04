import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
np.random.seed(42)  # 固定随机种子，保证结果可复现
# 实验参数
N = 15  # 小样本量
DATASET_PATH = "sinusoidal_dataset.npz"  # 数据集保存路径
M_LIST = [1, 3, 12]  # 多项式阶数

# ===================== 【核心】数据集生成 + 保存为NPZ =====================
def generate_and_save_dataset():
    """生成正弦拟合数据集，并单独保存为npz格式"""
    print("===== 正在生成并保存数据集 =====")
    # 生成原始数据
    x = np.random.uniform(0, 1, N)
    x = np.sort(x)  # 排序（绘图必需）
    y_true_sin = np.sin(2 * np.pi * x)
    epsilon = np.random.normal(0, 0.1, N)
    y_sin = y_true_sin + epsilon

    # 保存所有数组到NPZ文件（压缩存储）
    np.savez_compressed(
        DATASET_PATH,
        x=x,                # 排序后的输入特征
        y_sin=y_sin,        # 带噪声的标签
        y_true_sin=y_true_sin  # 真实正弦值
    )
    print(f"✅ 数据集已单独保存至：{DATASET_PATH}")
    print(f"【步骤A】正弦数据集x形状: {x.shape}，带噪声标签y形状: {y_sin.shape}")
    return x, y_sin, y_true_sin

# ===================== 【核心】加载已保存的NPZ数据集 =====================
def load_dataset():
    """从本地加载预保存的数据集，无需重复生成"""
    print("===== 正在加载本地数据集 =====")
    data = np.load(DATASET_PATH)
    x = data['x']
    y_sin = data['y_sin']
    y_true_sin = data['y_true_sin']
    print(f"✅ 成功加载数据集，样本量: {len(x)}")
    print(f"【步骤A】正弦数据集x形状: {x.shape}，带噪声标签y形状: {y_sin.shape}")
    return x, y_sin, y_true_sin

# ===================== 工具函数：构建多项式基设计矩阵（原代码保留） =====================
def build_poly_phi(x, M):
    """
    构建多项式基函数设计矩阵，第一列为偏置项
    :param x: 输入向量，形状(N,)
    :param M: 多项式阶数
    :return: Φ: 设计矩阵，形状(N, M+1)
    """
    N_sample = len(x)
    Φ = np.ones((N_sample, M + 1))
    for m in range(1, M + 1):
        Φ[:, m] = x ** m
    return Φ

# ===================== 主程序：选择 生成新数据 / 加载本地数据 =====================
# 首次运行：生成并保存数据 | 后续运行：注释下一行，取消注释加载行
x, y_sin, y_true_sin = generate_and_save_dataset()
# x, y_sin, y_true_sin = load_dataset()

# ===================== 步骤B+C：模型求解 + 可视化（原代码完全保留） =====================
x_dense = np.linspace(0, 1, 1000)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, M in enumerate(M_LIST):
    # 构建设计矩阵 + 满秩验证
    Φ = build_poly_phi(x, M)
    Φ_T_Φ = Φ.T @ Φ
    rank_Φ = np.linalg.matrix_rank(Φ_T_Φ)
    is_full_rank = rank_Φ == Φ.shape[1]
    print(f"【步骤B】M={M}时，Φ形状: {Φ.shape}，Φ^TΦ的秩: {rank_Φ}，是否满秩: {is_full_rank}")

    # 正规方程求解权重
    w_hat_poly = np.linalg.inv(Φ_T_Φ) @ Φ.T @ y_sin

    # 生成平滑预测曲线
    Φ_dense = build_poly_phi(x_dense, M)
    y_hat_poly = Φ_dense @ w_hat_poly

    # 绘图
    ax = axes[idx]
    ax.scatter(x, y_sin, c='blue', marker='o', s=60, label='Noisy Data Points')
    ax.plot(x_dense, np.sin(2 * np.pi * x_dense), 'green', linewidth=2, label='True Function sin(2πx)')
    ax.plot(x_dense, y_hat_poly, 'red', linewidth=2, label=f'Fitted Curve (M={M})')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Polynomial Order M={M}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)

plt.suptitle('Comparison of Sinusoidal Signal Fitting with Different Polynomial Orders', fontsize=16)
plt.savefig('Experiment2_Polynomial_Fitting_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n【实验2完成】所有步骤执行完毕！")