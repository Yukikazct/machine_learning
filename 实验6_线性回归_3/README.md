# 机器学习上机实验06：线性回归-3

## 奇异矩阵、岭回归与SVD几何解释

---

## 一、实验内容

| 文件 | 内容 |
|---|---|
| `Singularity&Ridge.py` | 正规方程奇异失效 + 岭回归修复（条件数分析） |
| `condition_number_and_stability.py` | 条件数与数值稳定性研究 |
| `Geometry Interpretation of SVD.py` | SVD 分解的几何意义可视化 |

---

## 二、实验要点

1. **矩阵奇异性**：当设计矩阵 $X^TX$ 奇异时，正规方程 $(X^TX)^{-1}X^Ty$ 无解
2. **岭回归**：添加 $\lambda I$ 使 $(X^TX + \lambda I)$ 可逆，解决奇异性问题
3. **条件数**：衡量矩阵对数值误差的敏感程度，条件数大则不稳定
4. **SVD 几何解释**：任意矩阵可分解为旋转→缩放→旋转的复合变换

---

## 三、文件结构

| 文件 | 说明 |
|---|---|
| `Singularity&Ridge.py` | 奇异矩阵 + 岭回归实验 |
| `condition_number_and_stability.py` | 条件数与数值稳定性 |
| `Geometry Interpretation of SVD.py` | SVD 几何解释 |
| `机器学习上机实验06：线性回归-3.pdf` | 实验指导文档 |

---

## 四、运行方式

```bash
python "Singularity&Ridge.py"
python condition_number_and_stability.py
python "Geometry Interpretation of SVD.py"
```
