import numpy as np

data = np.load('gmm_data.npz')
data_A = data['data_A']
data_B = data['data_B']

# ====================== 1. 一元MLE（簇A第一列） ======================
def mle_univariate(data):
    N = len(data)
    mu_hat = np.sum(data) / N
    sigma2_hat = np.sum((data - mu_hat) ** 2) / N
    return mu_hat, sigma2_hat

# 执行一元MLE
x_A1 = data_A[:, 0]
mu_mle_1d, sigma2_mle_1d = mle_univariate(x_A1)

# 打印一元结果
print('='*60)
print(' 一元MLE估计结果（簇A X1维度）')
print(f'估计均值：{mu_mle_1d:.4f} \t真实均值：1.0000')
print(f'估计方差：{sigma2_mle_1d:.4f} \t真实方差：1.0000')
print('='*60)

# ====================== 2. 二元MLE（正式实现：簇A + 簇B） ======================
def mle_bivariate(data):
    """
    二元高斯分布MLE估计
    :param data: 二维数据 (n_samples, 2)
    :return: 均值向量mu_hat, 协方差矩阵cov_hat
    """
    N = data.shape[0]
    # 估计均值向量
    mu_hat = np.mean(data, axis=0)
    # 估计协方差矩阵（严格按实验公式）
    diff = data - mu_hat
    cov_hat = np.dot(diff.T, diff) / N
    return mu_hat, cov_hat

# 对簇A执行二元MLE
mu_mle_A, cov_mle_A = mle_bivariate(data_A)
# 对簇B执行二元MLE
mu_mle_B, cov_mle_B = mle_bivariate(data_B)

# 打印簇A二元结果
print('\n 二元MLE估计结果（簇A）')
print(f'估计均值向量：{mu_mle_A.round(4)}')
print(f'真实均值向量：[1.0, 1.0]')
print(f'估计协方差矩阵：\n{cov_mle_A.round(4)}')
print(f'真实协方差矩阵：\n[[1.0, 0.8]\n [0.8, 1.0]]')

# 打印簇B二元结果
print('\n 二元MLE估计结果（簇B）')
print(f'估计均值向量：{mu_mle_B.round(4)}')
print(f'真实均值向量：[4.0, 4.0]')
print(f'估计协方差矩阵：\n{cov_mle_B.round(4)}')
print(f'真实协方差矩阵：\n[[1.2, -0.6]\n [-0.6, 0.8]]')
print('='*60)