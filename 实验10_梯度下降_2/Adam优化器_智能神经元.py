# 实验10：梯度下降-2 —— 手写Adam优化器：训练智能神经元
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# ========== 实验数据生成 ==========
np.random.seed(42)
class0 = np.random.randn(100, 2) + np.array([2, 2])
class1 = np.random.randn(100, 2) + np.array([6, 6])
X = np.vstack([class0, class1])
y = np.hstack([np.zeros(len(class0)), np.ones(len(class1))])

# 保存数据集
np.savez("dataset.npz", X=X, y=y, class0=class0, class1=class1)
print("数据集已保存：dataset.npz")


# ========== 任务1：mini-batch 数据生成器 ==========
def get_mini_batches(X, y, batch_size):
    """每个 epoch 开始前打乱数据，按 batch_size 切分返回数据流"""
    n = X.shape[0]
    indices = np.random.permutation(n)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    for i in range(0, n, batch_size):
        yield X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]


# ========== 任务2：前向传播与梯度计算 ==========
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(X_batch, w, b):
    """前向传播：线性组合 + sigmoid 激活"""
    z = X_batch @ w + b           # (batch, 2) @ (2,) + scalar → (batch,)
    y_pred = sigmoid(z)           # (batch,)
    return y_pred, z


def compute_loss(y_true, y_pred):
    """二分类交叉熵损失，加 epsilon 防止 log(0)"""
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def compute_gradients(X_batch, y_true, y_pred):
    """梯度：∂L/∂w = (ŷ-y)·x, ∂L/∂b = (ŷ-y)（矩阵化，按批平均）"""
    batch_size = X_batch.shape[0]
    error = y_pred - y_true                                      # (batch,)
    dw = (X_batch.T @ error) / batch_size                        # (2,)
    db = np.mean(error)
    return dw, db


# ========== 任务3：Adam 优化器权重更新 ==========
def adam_update(w, b, dw, db, m_w, v_w, m_b, v_b, t,
                lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """带偏差修正的 Adam 更新"""
    # 更新一阶矩和二阶矩
    m_w = beta1 * m_w + (1 - beta1) * dw
    v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
    m_b = beta1 * m_b + (1 - beta1) * db
    v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

    # 偏差修正
    m_w_hat = m_w / (1 - beta1 ** t)
    v_w_hat = v_w / (1 - beta2 ** t)
    m_b_hat = m_b / (1 - beta1 ** t)
    v_b_hat = v_b / (1 - beta2 ** t)

    # 参数更新
    w = w - lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
    b = b - lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    return w, b, m_w, v_w, m_b, v_b


# ========== 任务4：训练循环与可视化 ==========
def train(X, y, epochs, batch_size, lr=0.01, verbose=True):
    """完整训练循环，返回训练历史记录"""
    # 参数初始化
    w = np.random.randn(2) * 0.01
    b = 0.0

    # Adam 状态变量
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_b, v_b = 0.0, 0.0

    loss_history = []
    w_history = [w.copy()]  # 记录权重轨迹（用于画梯度下降路径）

    t = 0  # 全局时间步（跨 batch 累加）

    for epoch in range(epochs):
        epoch_losses = []
        for X_batch, y_batch in get_mini_batches(X, y, batch_size):
            t += 1
            y_pred, _ = forward(X_batch, w, b)
            loss = compute_loss(y_batch, y_pred)
            epoch_losses.append(loss * len(y_batch))  # 还原总损失用于加权平均
            dw, db = compute_gradients(X_batch, y_batch, y_pred)
            w, b, m_w, v_w, m_b, v_b = adam_update(
                w, b, dw, db, m_w, v_w, m_b, v_b, t, lr=lr
            )

        avg_loss = np.sum(epoch_losses) / len(y)
        loss_history.append(avg_loss)
        w_history.append(w.copy())

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch + 1:4d}/{epochs}  |  Loss: {avg_loss:.6f}")

    return w, b, np.array(loss_history), np.array(w_history)


def plot_results(loss_hist, w_hist, w_final, b_final, epochs, batch_size):
    """绘制：Loss曲线 + 决策边界 + 权重轨迹"""

    # 图1：loss 曲线
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(range(1, epochs + 1), loss_hist, marker='.', markersize=2, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title(f"Loss 曲线 (epochs={epochs}, batch={batch_size})")
    ax.grid(alpha=0.3)

    # 图2：决策边界
    ax = axes[1]
    w1, w2 = w_final
    ax.scatter(class0[:, 0], class0[:, 1], c="red", s=15, alpha=0.6, label="Class 0")
    ax.scatter(class1[:, 0], class1[:, 1], c="blue", s=15, alpha=0.6, label="Class 1")
    x_vals = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    if abs(w2) > 1e-8:
        y_vals = -(w1 * x_vals + b_final) / w2
        ax.plot(x_vals, y_vals, "k--", linewidth=1.5, label="Decision Boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"决策边界 (w=[{w1:.3f}, {w2:.3f}], b={b_final:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 图3：权重轨迹（误差平面上的梯度下降过程）
    ax = axes[2]
    # 在 w 空间内画出 loss 等高线
    w1_range = np.linspace(w_hist[:, 0].min() - 0.5, w_hist[:, 0].max() + 0.5, 200)
    w2_range = np.linspace(w_hist[:, 1].min() - 0.5, w_hist[:, 1].max() + 0.5, 200)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    # 使用当前偏置近似计算 loss 曲面
    loss_surface = np.zeros_like(W1)
    b_curr = b_final
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w_tmp = np.array([W1[i, j], W2[i, j]])
            y_pred, _ = forward(X, w_tmp, b_curr)
            loss_surface[i, j] = compute_loss(y, y_pred)

    levels = np.linspace(loss_surface.min(), loss_surface.max(), 30)
    ax.contour(W1, W2, loss_surface, levels=levels, cmap="viridis", alpha=0.7)
    ax.plot(w_hist[:, 0], w_hist[:, 1], 'r.-', markersize=3, linewidth=1, label="Weight trajectory")
    ax.scatter(w_hist[0, 0], w_hist[0, 1], c="green", s=60, marker="o", zorder=5, label="Start")
    ax.scatter(w_hist[-1, 0], w_hist[-1, 1], c="red", s=60, marker="*", zorder=5, label="End")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_title("误差平面上的权重轨迹")
    ax.legend(fontsize=8)

    plt.tight_layout()
    filename = f"results_e{epochs}_b{batch_size}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  结果图已保存：{filename}")
    plt.close(fig)


# ========== 主实验 ==========
if __name__ == "__main__":
    configs = [
        (20, 1),
        (20, 32),
        (100, 1),
        (100, 32),
    ]

    for epochs, batch_size in configs:
        print(f"\n{'='*60}")
        print(f"训练: epochs={epochs}, batch_size={batch_size}")
        print(f"{'='*60}")
        w_final, b_final, loss_hist, w_hist = train(
            X, y, epochs=epochs, batch_size=batch_size, lr=0.01
        )
        print(f"最终权重: w={w_final}, b={b_final:.6f}")
        print(f"最终 Loss: {loss_hist[-1]:.6f}")

        # 计算准确率
        y_pred_final, _ = forward(X, w_final, b_final)
        y_pred_label = (y_pred_final >= 0.5).astype(int)
        acc = np.mean(y_pred_label == y)
        print(f"训练集准确率: {acc:.4f}")

        plot_results(loss_hist, w_hist, w_final, b_final, epochs, batch_size)
