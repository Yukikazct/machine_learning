# 机器学习上机实验12：神经网络-2

## 基于PyTorch的深层神经网络训练机制研究

---

## 一、实验题目与目标

### 实验一：初始化方法对梯度传播的影响
- 对比 Normal 初始化与 Xavier 初始化在深层网络中的梯度传播行为
- 分析不同初始化方法对梯度消失/爆炸的影响

### 实验二：残差连接对深层网络训练的影响
- 对比普通 MLP 与残差 MLP 的训练特性
- 分析残差连接缓解梯度消失的机制

---

## 二、实验环境

| 组件 | 说明 |
|---|---|
| Python | 3.x |
| PyTorch | 2.x |
| Matplotlib | 绘图 |
| torchvision | MNIST 数据集 |

---

## 三、文件结构

| 文件 | 说明 |
|---|---|
| `experiment.py` | 完整实验代码（实验一 + 实验二） |
| `data/MNIST/` | MNIST 手写数字数据集 |
| `experiment_1_gradient_norm.png` | 实验一：初始化方法梯度范数对比 |
| `experiment_2_train_loss.png` | 实验二：训练 Loss 曲线 |
| `experiment_2_test_accuracy.png` | 实验二：测试准确率曲线 |
| `experiment_2_gradient_norm.png` | 实验二：残差连接梯度范数对比 |
| `机器学习上机实验12：神经网络-2.pdf` | 实验指导文档 |

---

## 四、运行方式

```bash
python experiment.py
```
