# 机器学习实验07-2：线性回归对离群点的鲁棒性缺陷
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 配置中文字体，避免绘图时标题或坐标轴出现乱码
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "Deja Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 全局随机种子
np.random.seed(42)

# 1. 构造基础数据
x0 = np.random.normal(2, 0.5, 100).reshape(-1, 1)  # 类别0
x1 = np.random.normal(3, 0.5, 100).reshape(-1, 1)  # 类别1
y0 = np.zeros(100)
y1 = np.ones(100)

# 基础数据集
x_base = np.vstack([x0, x1])
y_base = np.hstack([y0, y1])

# 含离群点数据集（x=20，标签1）
x_outlier = np.vstack([x_base, [[20]]])
y_outlier = np.hstack([y_base, [1]])


np.save('ols_x_base.npy', x_base)
np.save('ols_y_base.npy', y_base)
np.save('ols_x_outlier.npy', x_outlier)
np.save('ols_y_outlier.npy', y_outlier)


# 2. 计算决策边界
def get_decision_boundary(x, y):
    model = LinearRegression()
    model.fit(x, y)
    w = model.coef_[0]
    b = model.intercept_
    x_db = (0.5 - b) / w
    return x_db

db_base = get_decision_boundary(x_base, y_base)
db_out = get_decision_boundary(x_outlier, y_outlier)

# 3. 输出结果
print("\n==== 决策边界结果 ====")
print("无离群点决策边界：x =", round(db_base, 4))
print("有离群点决策边界：x =", round(db_out, 4))
print("决策边界偏移量：", round(abs(db_out - db_base), 4))

#  样本误伤检查
# 分类规则：x < 决策边界 → 类别0；x > 决策边界 → 类别1
correct_c1 = (x_base.ravel() > db_base) & (y_base == 1)
misclassified = correct_c1 & (x_base.ravel() < db_out)

# 统计误伤数量
mis_num = np.sum(misclassified)
total_c1 = np.sum(y_base == 1)
print("\n==== 样本误伤统计 ====")
print(f"类别1总样本数：{total_c1}")
print(f"被误分类为类别0的样本数：{mis_num}")
print(f"误伤比例：{mis_num/total_c1*100:.1f}%")
# ==================================================================

# 4. 绘图
plt.figure(figsize=(10, 5))
plt.scatter(x0, y0, c='blue', label='Class 0')
plt.scatter(x1, y1, c='red', label='Class 1')
# 高亮显示被误伤的样本
plt.scatter(x_base.ravel()[misclassified], y_base[misclassified],
            c='orange', s=150, edgecolors='black', label='被误伤的样本(错分为Class0)')
plt.scatter(20, 1, c='purple', s=100, marker='*', label='Outlier')

plt.axvline(db_base, color='green', linestyle='--', label=f'无离群点: {db_base:.2f}')
plt.axvline(db_out, color='orange', linestyle='--', label=f'有离群点: {db_out:.2f}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('OLS决策边界受离群点影响 + 样本误伤')
plt.legend()
plt.xlim(0, 22)
plt.show()