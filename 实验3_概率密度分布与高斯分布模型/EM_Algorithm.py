import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# 加载数据集
data = np.load('gmm_data.npz')
data_all = data['data_all']
data_A = data['data_A']  # 真实: 均值 [1,1]
data_B = data['data_B']  # 真实: 均值 [4,4]

# 1. 定义GMM-EM算法类
class GMM_EM:
    def __init__(self, K=2, tol=1e-6, max_iter=1000):
        self.K = K
        self.tol = tol
        self.max_iter = max_iter
        self.pi = None
        self.mu = None
        self.cov = None
        self.log_likelihoods = []

    # K-Means初始化
    def kmeans_init(self, X):
        n_samples = X.shape[0]
        kmeans = KMeans(n_clusters=self.K, random_state=42).fit(X)
        labels = kmeans.labels_
        self.pi = np.array([np.sum(labels == k) / n_samples for k in range(self.K)])
        self.mu = np.array([np.mean(X[labels == k], axis=0) for k in range(self.K)])
        self.cov = np.array([np.dot((X[labels == k] - self.mu[k]).T, (X[labels == k] - self.mu[k])) / np.sum(labels == k)
                             for k in range(self.K)])

    # E步
    def e_step(self, X):
        n_samples = X.shape[0]
        gamma = np.zeros((n_samples, self.K))
        for k in range(self.K):
            gamma[:, k] = self.pi[k] * multivariate_normal.pdf(X, mean=self.mu[k], cov=self.cov[k])
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma

    # M步
    def m_step(self, X, gamma):
        n_samples = X.shape[0]
        N_k = np.sum(gamma, axis=0)
        self.mu = np.dot(gamma.T, X) / N_k.reshape(-1, 1)
        for k in range(self.K):
            X_centered = X - self.mu[k]
            self.cov[k] = np.dot(gamma[:, k] * X_centered.T, X_centered) / N_k[k]
        self.pi = N_k / n_samples

    # 对数似然
    def calculate_log_likelihood(self, X):
        ll = 0
        for n in range(X.shape[0]):
            temp = 0
            for k in range(self.K):
                temp += self.pi[k] * multivariate_normal.pdf(X[n], mean=self.mu[k], cov=self.cov[k])
            ll += np.log(temp)
        return ll

    # 训练
    def fit(self, X):
        self.kmeans_init(X)
        init_ll = self.calculate_log_likelihood(X)
        self.log_likelihoods.append(init_ll)
        for iter in range(self.max_iter):
            gamma = self.e_step(X)
            self.m_step(X, gamma)
            new_ll = self.calculate_log_likelihood(X)
            self.log_likelihoods.append(new_ll)
            delta_ll = new_ll - self.log_likelihoods[-2]
            # 打印收敛时的迭代次数 + deltaL
            if delta_ll < self.tol:
                print(f'EM converged at iteration {iter+1}, deltaL = {delta_ll:.8f}')
                break
        if iter == self.max_iter - 1:
            print('Max iterations reached')
        return self.pi, self.mu, self.cov

# 训练模型
gmm_model = GMM_EM(K=2, tol=1e-6)
pi_final, mu_final, cov_final = gmm_model.fit(data_all)

# 修复：校正GMM分量顺序，对应正确的A/B簇
# 按照均值的大小排序，强制 mu1 → 簇A(1,1)，mu2 → 簇B(4,4)
order = np.argsort(mu_final[:, 0])  # 根据均值第一个元素升序排列
pi_final = pi_final[order]
mu_final = mu_final[order]
cov_final = cov_final[order]
# 把校正后的参数赋值回模型（保证可视化正确）
gmm_model.pi = pi_final
gmm_model.mu = mu_final
gmm_model.cov = cov_final

# 打印结果
print('\n' + '='*70)
print('GMM-EM Final Parameters')
print(f'Weights: π1(簇A)={pi_final[0]:.4f}, π2(簇B)={pi_final[1]:.4f}')
print(f'Mean μ1(簇A): {mu_final[0].round(4)}')
print(f'Mean μ2(簇B): {mu_final[1].round(4)}')
print(f'Covariance Σ1(簇A):\n{cov_final[0].round(4)}')
print(f'Covariance Σ2(簇B):\n{cov_final[1].round(4)}')
print('='*70)

# 保存参数
np.savez('gmm_em_final_params.npz', pi=pi_final, mu=mu_final, cov=cov_final, log_likelihoods=gmm_model.log_likelihoods)

# 可视化对比
kmeans_model = KMeans(n_clusters=2, random_state=42).fit(data_all)

def visualize_comparison():
    x_min, x_max = data_all[:, 0].min() - 1, data_all[:, 0].max() + 1
    y_min, y_max = data_all[:, 1].min() - 1, data_all[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    kmeans_pred = kmeans_model.predict(grid).reshape(xx.shape)
    gmm_ll = np.array([gmm_model.calculate_log_likelihood(grid[i:i+1]) for i in range(grid.shape[0])]).reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contour(xx, yy, kmeans_pred, levels=[0.5], colors='red', linewidths=2)
    plt.contour(xx, yy, gmm_ll, levels=10, colors='blue', alpha=0.7)
    plt.scatter(data_A[:, 0], data_A[:, 1], c='skyblue', label='Cluster A', alpha=0.7)
    plt.scatter(data_B[:, 0], data_B[:, 1], c='orange', label='Cluster B', alpha=0.7)
    plt.scatter(gmm_model.mu[:, 0], gmm_model.mu[:, 1], c='black', marker='*', s=250, label='GMM Center')
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], c='red', marker='o', s=120, label='K-Means Center')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('K-Means Hard Boundary vs GMM Probability Contour')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('kmeans_gmm_compare.png', dpi=300)
    plt.show()

def plot_log_likelihood():
    plt.figure(figsize=(8, 6))
    plt.plot(gmm_model.log_likelihoods, 'g-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iterations')
    plt.ylabel('Log Likelihood')
    plt.title('EM Algorithm Log Likelihood Curve')
    plt.grid(alpha=0.3)
    plt.savefig('em_log_likelihood.png', dpi=300)
    plt.show()

# 运行绘图
visualize_comparison()
plot_log_likelihood()