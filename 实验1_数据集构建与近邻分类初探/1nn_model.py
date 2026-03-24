# -*- coding: utf-8 -*-
"""
实验5：1-NN最近邻分类模型
任务要求：1. sklearn包实现 2. 自行编写代码实现（像素级距离计算）
运行方式：项目根目录执行 python code/1nn_model.py
前置依赖：已运行dataset_split.py完成实验4，生成训练/验证集npy
结果保存：data/processed/ 下生成2个预测结果npy（y_val_pred_sk/y_val_pred_man）
"""
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 路径自动配置（无需修改）
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
PROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'data/processed')
CLASS_LIST = ['building', 'mountain']  # 0=建筑，1=山脉

def load_split_data():
    """加载实验4拆分的训练集/验证集（测试集暂不使用）"""
    X_train = np.load(os.path.join(PROCESSED_ROOT, 'X_train.npy'))
    X_val = np.load(os.path.join(PROCESSED_ROOT, 'X_val.npy'))
    y_train = np.load(os.path.join(PROCESSED_ROOT, 'y_train.npy'))
    y_val = np.load(os.path.join(PROCESSED_ROOT, 'y_val.npy'))
    print(f" 加载实验4拆分数据完成")
    print(f"训练集：{X_train.shape} | 验证集：{X_val.shape}")
    return X_train, X_val, y_train, y_val

def euclidean_distance(x1, x2):
    """自行编写：欧氏距离计算（像素级比较核心）√Σ(x1-x2)²"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_sklearn_implement(X_train, X_val, y_train, y_val):
    """sklearn包实现1-NN模型（实验要求学习使用）"""
    # 初始化1-NN模型：欧氏距离
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    # KNN为懒学习，训练仅存储样本，无参数更新
    knn.fit(X_train, y_train)
    # 验证集预测
    y_val_pred = knn.predict(X_val)
    # 计算准确率
    acc = accuracy_score(y_val, y_val_pred)
    # 保存预测结果
    np.save(os.path.join(PROCESSED_ROOT, 'y_val_pred_sk.npy'), y_val_pred)
    # 打印结果
    print("\n===== 实验5：sklearn版1-NN模型结果（验证集） =====")
    print(f"1-NN验证集准确率：{acc:.4f} ({acc*100:.2f}%)")
    print(f"✅ 预测结果已保存：y_val_pred_sk.npy")
    return y_val_pred, acc

def knn_manual_implement(X_train, X_val, y_train, y_val):
    """自行编写代码实现1-NN模型（实验核心要求）"""
    y_val_pred = []
    # 遍历验证集每个样本，逐个找最近邻
    for x_val in X_val:
        # 计算与所有训练样本的欧氏距离
        distances = [euclidean_distance(x_val, x_train) for x_train in X_train]
        # 找到距离最小的训练样本索引
        min_dist_idx = np.argmin(distances)
        # 用最近邻标签作为预测标签
        y_val_pred.append(y_train[min_dist_idx])
    y_val_pred = np.array(y_val_pred)
    # 计算准确率
    acc = accuracy_score(y_val, y_val_pred)
    # 保存预测结果
    np.save(os.path.join(PROCESSED_ROOT, 'y_val_pred_man.npy'), y_val_pred)
    # 打印结果
    print("\n===== 实验5：手动编写版1-NN模型结果（验证集） =====")
    print(f"1-NN验证集准确率：{acc:.4f} ({acc*100:.2f}%)")
    print(f" 预测结果已保存：y_val_pred_man.npy")
    return y_val_pred, acc

if __name__ == '__main__':
    # 检查拆分数据是否存在
    if not os.path.exists(os.path.join(PROCESSED_ROOT, 'X_train.npy')):
        raise FileNotFoundError("请先运行dataset_split.py完成实验4，生成训练/验证集npy！")
    # 执行1-NN两种实现
    X_train, X_val, y_train, y_val = load_split_data()
    knn_sklearn_implement(X_train, X_val, y_train, y_val)
    knn_manual_implement(X_train, X_val, y_train, y_val)