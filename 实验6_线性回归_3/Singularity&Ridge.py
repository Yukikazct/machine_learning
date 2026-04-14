# 实验1：正规方程奇异失效 + 岭回归修复
import numpy as np
from numpy.linalg import inv

np.random.seed(42)

n_samples = 100
x1 = np.random.randn(n_samples)
x2 = np.random.randn(n_samples)
x3 = x1 + x2  # 第三列 = 前两列和（严格线性相关）
X = np.column_stack([x1, x2, x3])
w_true = np.array([1, 2, 3])
y = X @ w_true


X_T_X = X.T @ X

# 1. 正规方程
try:
    w_normal = inv(X_T_X) @ X.T @ y
except np.linalg.LinAlgError:

    w_normal = np.linalg.pinv(X_T_X) @ X.T @ y

# 2. 岭回归
lambda_ridge = 0.1
w_ridge = inv(X_T_X + lambda_ridge * np.eye(3)) @ X.T @ y

# 输出结果
print("真实系数：", w_true)
print("正规方程解：", w_normal)
print("岭回归解：", w_ridge)
print("正规方程偏差：", np.linalg.norm(w_normal - w_true))
print("岭回归偏差：", np.linalg.norm(w_ridge - w_true))