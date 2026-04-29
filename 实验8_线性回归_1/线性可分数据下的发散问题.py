# 实验2：线性可分数据下的参数发散与L2正则化
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

def cross_entropy_loss(y, y_hat):
    """交叉熵损失（防log(0)报错）"""
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def l2_reg_loss(y, x, w, lam):
    """带L2正则的交叉熵损失"""
    y_hat = sigmoid(w * x)
    ce = cross_entropy_loss(y, y_hat)
    reg = 0.5 * lam * (w ** 2)
    return ce + reg

# 2. 线性可分实验数据
x = np.array([-4, -3, -2, 1, 2, 5])
y = np.array([0, 0, 0, 1, 1, 1])

# 3. 无正则化：损失曲线（w发散）
w_no_reg = np.linspace(-10, 5, 100)
loss_no_reg = []
for w in w_no_reg:
    y_hat = sigmoid(w * x)
    loss_no_reg.append(cross_entropy_loss(y, y_hat))

# 4. L2正则化：不同λ对比
lam_list = [0, 0.01, 0.1, 1, 10]
w_reg = np.linspace(-10, 10, 100)
loss_reg = {lam: [] for lam in lam_list}

for lam in lam_list:
    for w in w_reg:
        loss_reg[lam].append(l2_reg_loss(y, x, w, lam))

# 5. 绘图可视化
plt.figure(figsize=(12, 5))

# 无正则化：参数发散
plt.subplot(1, 2, 1)
plt.plot(w_no_reg, loss_no_reg, 'g-', linewidth=2, label='无正则 CE Loss')
plt.title('线性可分-无正则：w→∞，损失持续下降')
plt.xlabel('w')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# L2正则化：参数收敛
plt.subplot(1, 2, 2)
for lam in lam_list:
    plt.plot(w_reg, loss_reg[lam], linewidth=2, label=f'λ={lam}')
plt.title('L2正则化：λ越大，最优w越靠近0')
plt.xlabel('w')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.suptitle('实验2：线性可分与正则化')
plt.show()