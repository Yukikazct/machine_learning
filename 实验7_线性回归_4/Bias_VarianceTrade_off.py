import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

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

# 全局随机种子
np.random.seed(42)


# 真实函数
def true_func(x):
    return np.sin(2 * np.pi * x)


# 1. 测试集（无噪声）
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_true = true_func(x_test)

# 2. 训练集参数
n_datasets = 50
n_samples = 20
sigma = 0.15

all_train_datasets = []

# 3. 多项式特征（10阶）
poly = PolynomialFeatures(degree=10)
x_test_poly = poly.fit_transform(x_test)

# 4. 正则化参数
lambdas = [1e-8, 1e-7, 1e-6, 1e-3, 0.01, 0.1, 1, 10]
y_pred_all = []

# 批量拟合
for lam in lambdas:
    y_pred_per_lam = []
    for _ in range(n_datasets):
        # 生成训练集x：均匀随机采样
        x_train = np.random.uniform(0, 1, n_samples).reshape(-1, 1)
        # 噪声分布：N(0, 0.15²)，即均值0，标准差sigma=0.15的高斯噪声
        y_train = true_func(x_train) + np.random.normal(0, sigma, n_samples).reshape(-1, 1)

        # 保存每组训练数据
        if lam == lambdas[0]:
            all_train_datasets.append((x_train, y_train))

        # 多项式特征转换
        x_train_poly = poly.fit_transform(x_train)
        # 岭回归训练
        model = Ridge(alpha=lam, random_state=42)
        model.fit(x_train_poly, y_train)
        # 预测
        y_pred = model.predict(x_test_poly)
        y_pred_per_lam.append(y_pred.ravel())
    y_pred_all.append(np.array(y_pred_per_lam))


np.save('linear_regression_train_datasets.npy', all_train_datasets)


# 5. 计算偏差²、方差、总误差
bias2_list = []
var_list = []
total_error_list = []
noise_var = sigma ** 2

for i in range(len(lambdas)):
    y_pred = y_pred_all[i]
    y_pred_mean = np.mean(y_pred, axis=0)
    bias2 = np.mean((y_pred_mean - y_true.ravel()) ** 2)
    var = np.mean(np.var(y_pred, axis=0))
    total_error = np.mean((y_pred - y_true.ravel()) ** 2)
    bias2_list.append(bias2)
    var_list.append(var)
    total_error_list.append(total_error)

# 打印结果
print("\n噪声方差：", round(noise_var, 4))
print("λ\tBias²\t\tVar\t\tTotalError")
for i, lam in enumerate(lambdas):
    print(f"{lam:.1e}\t{bias2_list[i]:.4f}\t{var_list[i]:.4f}\t{total_error_list[i]:.4f}")

# 6. 绘图：偏差方差权衡
plt.figure(figsize=(10, 5))
log_lam = np.log10(lambdas)
plt.plot(log_lam, bias2_list, 'r-o', label='Bias²')
plt.plot(log_lam, var_list, 'g-o', label='Variance')
plt.plot(log_lam, total_error_list, 'b-o', label='Total Error')
plt.xlabel('log10(λ)')
plt.ylabel('Error')
plt.title('Bias-Variance Trade-off')
plt.legend()
plt.grid(True)
plt.show()

# 7. 极端λ与最优λ曲线对比
best_idx = np.argmin(total_error_list)
best_lam = lambdas[best_idx]
plot_lams = [1e-8, best_lam, 10]
plot_names = [f"λ=1e-8(过拟合)", f"λ={best_lam:.2e}(最优)", f"λ=10(欠拟合)"]

plt.figure(figsize=(15, 5))
for i, (lam, name) in enumerate(zip(plot_lams, plot_names)):
    idx = lambdas.index(lam)
    y_pred = y_pred_all[idx]
    plt.subplot(1, 3, i + 1)
    plt.plot(x_test, y_true, 'r-', label='True sin(2πx)')
    for yp in y_pred:
        plt.plot(x_test, yp, 'c-', alpha=0.3)
    plt.plot(x_test, np.mean(y_pred, axis=0), 'b-', label='Mean Pred')
    plt.title(name)
    plt.legend()
    plt.ylim(-1.5, 1.5)
plt.tight_layout()
plt.show()