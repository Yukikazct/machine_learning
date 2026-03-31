import numpy as np
import matplotlib.pyplot as plt

# 全局固定随机种子
np.random.seed(42)

# 步骤A：合成正弦数据集
N = 15  # 小样本量，易观察过拟合（实验手册指定）
x = np.random.uniform(0, 1, N)  # 输入x∈[0,1]均匀分布
x = np.sort(x)  # 排序方便后续绘图
# 真实函数：y=sin(2πx)，加入高斯噪声ε~N(0,0.1)
y_true_sin = np.sin(2 * np.pi * x)
epsilon = np.random.normal(0, 0.1, N)
y_sin = y_true_sin + epsilon
print(f"【步骤A】正弦数据集x形状: {x.shape}，带噪声标签y形状: {y_sin.shape}")


# 工具函数：构建多项式基设计矩阵Φ
def build_poly_phi(x, M):
    """
    构建多项式基函数设计矩阵，满足实验手册要求：第一列为偏置项
    :param x: 输入向量，形状(N,)
    :param M: 多项式阶数
    :return: Φ: 设计矩阵，形状(N, M+1)，列依次为[1, x, x², ..., x^M]
    """
    N_sample = len(x)
    Φ = np.ones((N_sample, M + 1))  # 第一列全1（偏置）
    for m in range(1, M + 1):
        Φ[:, m] = x ** m  # 后续列为x的各阶幂次
    return Φ


# 步骤B+C：基矩阵构建+满秩验证+多阶模型求解+可视化
M_list = [1, 3, 12]  # 实验手册指定待测试阶数
# 生成密集x轴，用于绘制平滑的预测曲线（提升可视化效果）
x_dense = np.linspace(0, 1, 1000)
# 创建画布：1行3列子图，对比不同阶数拟合效果
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 遍历不同阶数，逐一求解并可视化
for idx, M in enumerate(M_list):
    # 步骤B：构建当前阶数的多项式基设计矩阵Φ
    Φ = build_poly_phi(x, M)
    # 步骤B：验证Φ^TΦ的满秩性
    Φ_T_Φ = Φ.T @ Φ
    rank_Φ = np.linalg.matrix_rank(Φ_T_Φ)
    is_full_rank = rank_Φ == Φ.shape[1]
    print(f"【步骤B】M={M}时，Φ形状: {Φ.shape}，Φ^TΦ的秩: {rank_Φ}，是否满秩: {is_full_rank}")

    # 步骤C：正规方程求解最优权重w_hat
    w_hat_poly = np.linalg.inv(Φ_T_Φ) @ Φ.T @ y_sin

    # 步骤C：对密集x预测，生成平滑拟合曲线
    Φ_dense = build_poly_phi(x_dense, M)
    y_hat_poly = Φ_dense @ w_hat_poly

    ax = axes[idx]
    ax.scatter(x, y_sin, c='blue', marker='o', s=60, label='Noisy Data Points')
    ax.plot(x_dense, np.sin(2 * np.pi * x_dense), 'green', linewidth=2, label='True Function sin(2πx)')
    ax.plot(x_dense, y_hat_poly, 'red', linewidth=2, label=f'Fitted Curve (M={M})')
    # Subplot style settings
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