# 机器学习上机实验03：概率密度估计与高斯混合模型

## MLE、EM算法与GMM实现

---

## 一、实验内容

1. **数据生成**：生成异质高斯分布数据集（两个簇，不同协方差结构）
2. **MLE 参数估计**：使用极大似然估计法拟合单高斯分布
3. **EM 算法**：从零实现 EM 算法求解高斯混合模型（GMM）
4. **K-Means vs GMM**：对比 K-Means 硬聚类与 GMM 软聚类的差异

---

## 二、文件结构

### 代码

| 文件 | 说明 |
|---|---|
| `data generation.py` | 异质高斯数据生成（簇A: 200 点正相关，簇B: 400 点负相关） |
| `MLE.py` | 极大似然估计（单高斯模型参数估计） |
| `EM_Algorithm.py` | EM 算法求解 GMM + K-Means 对比 |

### 数据与结果

| 文件 | 说明 |
|---|---|
| `gmm_data.npz` | 生成的高斯混合数据 |
| `gmm_em_final_params.npz` | EM 算法最终估计参数 |
| `data_generation.png` | 数据分布可视化 |
| `em_log_likelihood.png` | EM 迭代对数似然曲线 |
| `kmeans_gmm_compare.png` | K-Means vs GMM 聚类对比 |

### 其他

| 文件 | 说明 |
|---|---|
| `机器学习上机实验03：概率密度估计与高斯混合模型.pdf` | 实验指导文档 |

---

## 三、运行方式

```bash
python "data generation.py"   # 生成数据
python MLE.py                 # 极大似然估计
python EM_Algorithm.py        # EM算法 + GMM
```
