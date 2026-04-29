# 实验1：逻辑回归损失函数几何性质（MSE与交叉熵对比）
import numpy as np
import matplotlib.pyplot as plt


# 配置中文字体，避免绘图时标题或坐标轴出现乱码
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# 1. 核心函数定义
def sigmoid(z):
    """sigmoid激活函数"""
    return 1 / (1 + np.exp(-z))

def mse_loss(y, y_hat):
    """均方误差损失"""
    return 0.5 * np.sum((y - y_hat) ** 2)

def cross_entropy_loss(y, y_hat):
    """交叉熵损失（防log(0)报错）"""
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# 2. 实验数据
x = np.array([-4.0, -2.0, -0.5, 1.0, 3.0, 5.0])
y = np.array([0, 0, 1, 0, 1, 1])

# ===================== 保存数据集到本地 =====================
np.save("exp08_task1_x.npy", x)    # 保存特征x
np.save("exp08_task1_y.npy", y)    # 保存标签y
print("实验1数据集已保存：exp08_task1_x.npy、exp08_task1_y.npy")

# 3. 遍历w计算损失
w_list = np.linspace(-10, 10, 100)
mse_losses = []
ce_losses = []

for w in w_list:
    y_hat = sigmoid(w * x)
    mse_losses.append(mse_loss(y, y_hat))
    ce_losses.append(cross_entropy_loss(y, y_hat))

# 4. 绘图可视化
plt.figure(figsize=(12, 5))

# MSE损失曲线
plt.subplot(1, 2, 1)
plt.plot(w_list, mse_losses, 'r-', linewidth=2, label='MSE Loss')
plt.title('MSE Loss 曲线')
plt.xlabel('w')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 交叉熵损失曲线
plt.subplot(1, 2, 2)
plt.plot(w_list, ce_losses, 'b-', linewidth=2, label='Cross-Entropy Loss')
plt.title('Cross-Entropy Loss 曲线')
plt.xlabel('w')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.suptitle('实验1：损失函数几何性质')
plt.show()