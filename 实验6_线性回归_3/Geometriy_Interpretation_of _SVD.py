# 实验2：SVD 几何可视化
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

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

theta = np.linspace(0, 2 * np.pi, 100)
unit_circle = np.array([np.cos(theta), np.sin(theta)])  # 2×100
start_point = np.array([[1], [0]])  # 起始红点

# 随机2×2矩阵做SVD
A = np.array([[2, 1], [1, 2]])
U, Sigma, V_T = svd(A)
Sigma_mat = np.diag(Sigma)

# 分步变换
circle_vt = V_T @ unit_circle
circle_sigma = Sigma_mat @ circle_vt
circle_u = U @ circle_sigma

# 红点轨迹
p_vt = V_T @ start_point
p_sigma = Sigma_mat @ p_vt
p_u = U @ p_sigma

# 绘图
plt.figure(figsize=(12, 3))
plt.subplot(141)
plt.plot(unit_circle[0], unit_circle[1], 'b-')
plt.scatter(start_point[0], start_point[1], c='r', s=50)
plt.title('原始单位圆')
plt.axis('equal')

plt.subplot(142)
plt.plot(circle_vt[0], circle_vt[1], 'g-')
plt.scatter(p_vt[0], p_vt[1], c='r', s=50)
plt.title('V^T 旋转')
plt.axis('equal')

plt.subplot(143)
plt.plot(circle_sigma[0], circle_sigma[1], 'orange')
plt.scatter(p_sigma[0], p_sigma[1], c='r', s=50)
plt.title('Σ 缩放')
plt.axis('equal')

plt.subplot(144)
plt.plot(circle_u[0], circle_u[1], 'purple')
plt.scatter(p_u[0], p_u[1], c='r', s=50)
plt.title('U 旋转')
plt.axis('equal')

plt.tight_layout()
plt.show()



print(f"原始点：{start_point.flatten()}")
print(f"V^T 后：{p_vt.flatten()}")
print(f"Σ 后：{p_sigma.flatten()}")
print(f"U 后：{p_u.flatten()}")