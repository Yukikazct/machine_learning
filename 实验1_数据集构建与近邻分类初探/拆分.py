# -*- coding: utf-8 -*-
"""
实验4：数据集拆分
任务要求：70%训练集 / 15%验证集 / 15%测试集，科学分层拆分（保证类别比例）
运行方式：项目根目录执行 python code/dataset_split.py
结果保存：data/processed/ 下生成6个npy文件（X_train/X_val/X_test/y_train/y_val/y_test）
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# 路径自动配置（与前序代码一致，无需修改）
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
PROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'data/processed')
SEED = 42  # 固定随机种子，结果可复现
CLASS_LIST = ['building', 'mountain']  # 0=建筑，1=山脉


def load_and_flatten():
    """加载预处理后的特征，展平为1维（KNN算法要求）"""
    features = np.load(os.path.join(PROCESSED_ROOT, 'features.npy'))
    labels = np.load(os.path.join(PROCESSED_ROOT, 'labels.npy'))
    # 64x64x3 → 12288维，展平特征
    features_flat = features.reshape(features.shape[0], -1)
    print(f"✅ 加载数据完成：总样本{len(labels)} | 建筑(0)={np.sum(labels == 0)} | 山脉(1)={np.sum(labels == 1)}")
    print(f"✅ 特征展平：{features.shape} → {features_flat.shape}")
    return features_flat, labels


def split_data(X, y):
    """分层拆分数据集：70%训练 → 30%临时集再拆分为15%验证+15%测试"""
    # 第一步：拆分为训练集（70%）和临时集（30%）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y  # stratify=分层抽样核心
    )
    # 第二步：临时集拆分为验证集（15%）和测试集（15%）
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    # 保存拆分结果到data/processed
    np.save(os.path.join(PROCESSED_ROOT, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_ROOT, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_ROOT, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_ROOT, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_ROOT, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_ROOT, 'y_test.npy'), y_test)

    # 打印拆分结果（实验报告可直接复制）
    print("\n===== 实验4：数据集70%/15%/15%拆分结果 =====")
    print(f"训练集：{X_train.shape[0]}样本 | 建筑(0)={np.sum(y_train == 0)} | 山脉(1)={np.sum(y_train == 1)}")
    print(f"验证集：{X_val.shape[0]}样本 | 建筑(0)={np.sum(y_val == 0)} | 山脉(1)={np.sum(y_val == 1)}")
    print(f"测试集：{X_test.shape[0]}样本 | 建筑(0)={np.sum(y_test == 0)} | 山脉(1)={np.sum(y_test == 1)}")
    print(f"\n✅ 拆分结果已保存至：{PROCESSED_ROOT}")
    print(f"✅ 生成文件：X_train/X_val/X_test/y_train/y_val/y_test.npy")


if __name__ == '__main__':
    # 检查预处理数据是否存在
    if not os.path.exists(os.path.join(PROCESSED_ROOT, 'features.npy')):
        raise FileNotFoundError("请先运行data_process.py完成实验2+3，生成features.npy和labels.npy！")
    # 执行拆分
    X, y = load_and_flatten()
    split_data(X, y)