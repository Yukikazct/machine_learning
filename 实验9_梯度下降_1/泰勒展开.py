# 任务1：泰勒展开的“局部逼近”验证
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 定义原函数
def f_original(x, y):
    return np.sin(x) * np.cos(y)

# 定义二阶泰勒近似函数
def f_taylor(x, y):
    x0 = np.pi / 2
    y0 = 0
    return 1 - 0.5 * (x - x0)**2 - 0.5 * y**2

# 生成网格数据
x = np.linspace(np.pi/2 - 1.5, np.pi/2 + 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
Z_original = f_original(X, Y)
Z_taylor = f_taylor(X, Y)

# 保存数据
np.savez("task1_taylor_data.npz", X=X, Y=Y, Z_original=Z_original, Z_taylor=Z_taylor, expand_point=[np.pi/2, 0])
print("任务1数据已保存：task1_taylor_data.npz")

# 3D绘图
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_original, cmap="viridis", alpha=0.8)
ax1.set_title("原函数 f(x,y)=sin(x)cos(y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x,y)")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_taylor, cmap="plasma", alpha=0.8)
ax2.set_title("二阶泰勒近似函数")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f̂(x,y)")

plt.tight_layout()
plt.show()