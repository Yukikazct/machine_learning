# 实验3：病态矩阵 + 条件数 + 噪声敏感度
import numpy as np
from numpy.linalg import inv, cond
from sklearn.linear_model import Ridge



np.random.seed(42)

n_samples = 100
x1 = np.random.randn(n_samples)
x2 = np.random.randn(n_samples)
w_true = np.array([1, 2, 3])

eta = 1e-8
x3_ill = x1 + x2 + eta * np.random.randn(n_samples)  # 加极小扰动
X_ill = np.column_stack([x1, x2, x3_ill])
X_ill_T_X_ill = X_ill.T @ X_ill
cond_num = cond(X_ill_T_X_ill)


y_clean = X_ill @ w_true
noise = 1e-4 * np.random.randn(n_samples)
y_noisy = y_clean + noise

# 正规方程求解
w_clean = inv(X_ill_T_X_ill) @ X_ill.T @ y_clean
w_noisy = inv(X_ill_T_X_ill) @ X_ill.T @ y_noisy


lambda_ridge = 0.1
ridge = Ridge(alpha=lambda_ridge, fit_intercept=False)
ridge.fit(X_ill, y_noisy)
w_ridge_ill = ridge.coef_

# 输出结果
print(f"病态矩阵 X^T X 条件数：{cond_num:.2e}")
print("无噪声正规方程解：", w_clean)
print("加噪声正规方程解：", w_noisy)
print("解波动幅度：", np.linalg.norm(w_noisy - w_clean))
print("岭回归解：", w_ridge_ill)
print("岭回归偏差：", np.linalg.norm(w_ridge_ill - w_true))