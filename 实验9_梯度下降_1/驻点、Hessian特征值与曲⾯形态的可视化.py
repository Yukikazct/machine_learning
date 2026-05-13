# 任务2：驻点、Hessian特征值与曲面形态可视化 - 独立完整代码
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# 定义三个函数
def f1(x, y):
    return x**2 + y**2

def f2(x, y):
    return -x**2 - y**2

def f3(x, y):
    return x**2 - y**2

# 定义Hessian矩阵
H1 = np.array([[2, 0], [0, 2]])
H2 = np.array([[-2, 0], [0, -2]])
H3 = np.array([[2, 0], [0, -2]])

# 计算特征值
eig1 = np.linalg.eigvals(H1)
eig2 = np.linalg.eigvals(H2)
eig3 = np.linalg.eigvals(H3)

# 生成网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z1 = f1(X, Y)
Z2 = f2(X, Y)
Z3 = f3(X, Y)

# 保存数据
np.savez("task2_hessian_data.npz", X=X, Y=Y, Z1=Z1, Z2=Z2, Z3=Z3, H1=H1, H2=H2, H3=H3, eig1=eig1, eig2=eig2, eig3=eig3)
print("任务2数据已保存：task2_hessian_data.npz")
print(f"f1特征值：{eig1} → 局部极小值")
print(f"f2特征值：{eig2} → 局部极大值")
print(f"f3特征值：{eig3} → 鞍点")

# 3D绘图
fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap="viridis")
ax1.set_title("f1=x^2+y^2（局部极小值）")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f1")

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap="plasma")
ax2.set_title("f2=-x^2-y^2（局部极大值）")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f2")

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z3, cmap="coolwarm")
ax3.set_title("f3=x^2-y^2（鞍点）")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("f3")

plt.tight_layout()
plt.show()