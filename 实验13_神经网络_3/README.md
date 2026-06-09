# 机器学习上机实验13：神经网络-3

## 基于PyTorch的Transformer Encoder图像分类

---

## 一、实验内容

| 任务 | 内容 |
|---|---|
| 任务1 | 数据加载 — MNIST 数据集 |
| 任务2 | 构建 TransformerClassifier |
| 任务3 | 完成 Forward 函数实现 |
| 任务4 | 模型训练 |
| 任务5 | 查看模型结构 |
| 任务6 | Head 数量实验（nhead=1, 2, 4, 8） |
| 任务7 | Position Encoding 消融实验 |

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
| `experiment.py` | 完整实验代码 |
| `data/MNIST/` | MNIST 手写数字数据集 |
| `experiment_head_count.png` | 不同 Head 数量的准确率对比 |
| `experiment_pe_ablation.png` | Position Encoding 消融实验结果 |
| `机器学习上机实验13-14：神经网络-3.pdf` | 实验指导文档 |

---

## 四、运行方式

```bash
python experiment.py
```
