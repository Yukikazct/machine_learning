# 任务3：等误差线、特征向量与轴向对齐 - 独立完整代码
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# 定义Hessianhttps://www.speedtest.cn矩阵
H = np.array([[3, 2], [2, 3]])

# 计算特征值与特征向量
eig_vals, eig_vecs = np.linalg.eig(H)
print(f"Hessian特征值：{eig_vals}")
print(f"Hessian特征向量：\n{eig_vecs}")

# 定义二次型函数
def quadratic_form(x, y, H):
    w = np.array([x, y])
    return 0.5 * w.T @ H @ w

# 生成网格
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = quadratic_form(X[i, j], Y[i, j], H)

# 保存数据
np.savez("task3_contour_data.npz", X=X, Y=Y, Z=Z, H=H, eig_vals=eig_vals, eig_vecs=eig_vecs)
print("任务3数据已保存：task3_contour_data.npz")

# 等高线+特征向量绘图
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=20, cmap="viridis")
plt.clabel(contour, inline=True, fontsize=8)

# 绘制特征向量
plt.quiver(0, 0, eig_vecs[0,0], eig_vecs[1,0], color="red", scale=5, label=f"λ={eig_vals[0]:.2f}")
plt.quiver(0, 0, eig_vecs[0,1], eig_vecs[1,1], color="blue", scale=5, label=f"λ={eig_vals[1]:.2f}")

plt.title("等误差线与Hessian特征向量")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
plt.legend()
plt.grid(alpha=0.3)
plt.axis("equal")
plt.show()