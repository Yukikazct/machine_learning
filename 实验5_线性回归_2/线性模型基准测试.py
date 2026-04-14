import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# -------------------------- 全局配置 --------------------------
DATA_ROOT = "data"
IMG_SIZE = 32
LAMBDA_BASE = 1e-6
LAMBDA_LIST = [1e-6, 1e-3, 1.0, 10.0, 100.0, 1000.0]


# -------------------------- 环节1依赖：数据加载函数 --------------------------
def load_data(data_root, img_size):
    train_X, train_y = [], []
    val_X, val_y = [], []
    for phase in ["train", "val"]:
        for label, cls in enumerate(["cat", "dog"]):
            cls_path = os.path.join(data_root, phase, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_resized = cv2.resize(img, (img_size, img_size))
                img_norm = img_resized / 255.0
                img_flat = img_norm.flatten()
                if phase == "train":
                    train_X.append(img_flat)
                    train_y.append(label)
                else:
                    val_X.append(img_flat)
                    val_y.append(label)
    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32).reshape(-1, 1)
    val_X = np.array(val_X, dtype=np.float32)
    val_y = np.array(val_y, dtype=np.float32).reshape(-1, 1)
    return train_X, train_y, val_X, val_y

# -------------------------- 环节2核心函数 --------------------------
def add_bias(X):
    """为特征矩阵添加偏置项（第一列全1），构建设计矩阵"""
    m = X.shape[0]
    bias = np.ones((m, 1))
    X_design = np.hstack((bias, X))
    return X_design

def linear_regression(X, y, lambda_):
    """正规方程求解岭回归参数：$\hat{w}=(X^T X+\lambda I)^{-1} X^T y$"""
    n = X.shape[1]
    X_T = X.T
    XTX = np.dot(X_T, X)
    lambda_I = lambda_ * np.eye(n)  # 防止XTX奇异
    XTX_lambda = XTX + lambda_I
    XTX_inv = np.linalg.inv(XTX_lambda)
    w = np.dot(np.dot(XTX_inv, X_T), y)
    return w

def predict(X_design, w, threshold=0.5):
    """预测函数：阈值0.5二分类（>0.5=狗1，<0.5=猫0）"""
    y_pred = np.dot(X_design, w)
    y_pred_label = (y_pred > threshold).astype(np.float32)
    return y_pred_label

# -------------------------- 执行运行 --------------------------
if __name__ == "__main__":
    # 1. 加载预处理数据
    train_X, train_y, val_X, val_y = load_data(DATA_ROOT, IMG_SIZE)
    # 2. 构建设计矩阵（加偏置项，1025维）
    train_X1 = add_bias(train_X)
    val_X1 = add_bias(val_X)
    # 3. 正规方程求解权重（极小正则化λ=1e-6）
    w1_base = linear_regression(train_X1, train_y, LAMBDA_BASE)
    # 4. 预测并计算准确率
    train_y_pred1 = predict(train_X1, w1_base)
    val_y_pred1 = predict(val_X1, w1_base)
    acc_train1 = accuracy_score(train_y, train_y_pred1)
    acc_val1 = accuracy_score(val_y, val_y_pred1)
    # 5. 输出结果
    print("="*60)
    print("【环节2-线性模型基准测试】执行完成")
    print(f"一次项线性模型 - 训练集准确率：{acc_train1:.4f}")
    print(f"一次项线性模型 - 验证集准确率：{acc_val1:.4f}")
    print(f"训练/验证集准确率差异：{abs(acc_train1 - acc_val1):.4f}")
    print(f"权重向量维度：{w1_base.shape}（1025维：1024特征+1偏置）")
    print("="*60)