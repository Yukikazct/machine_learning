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

# -------------------------- 环节1/2依赖函数 --------------------------
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

def add_bias(X):
    """添加偏置项"""
    m = X.shape[0]
    bias = np.ones((m, 1))
    X_design = np.hstack((bias, X))
    return X_design

def linear_regression(X, y, lambda_):
    """正规方程求解权重"""
    n = X.shape[1]
    X_T = X.T
    XTX = np.dot(X_T, X)
    lambda_I = lambda_ * np.eye(n)
    XTX_lambda = XTX + lambda_I
    XTX_inv = np.linalg.inv(XTX_lambda)
    w = np.dot(np.dot(XTX_inv, X_T), y)
    return w

def predict(X_design, w, threshold=0.5):
    """阈值二分类预测"""
    y_pred = np.dot(X_design, w)
    y_pred_label = (y_pred > threshold).astype(np.float32)
    return y_pred_label

# -------------------------- 环节3核心函数 --------------------------
def poly2_feature(X):
    """
    手动实现二次项（平方项）特征映射
    输出：m×2048维（1024一次项+1024平方项），加偏置后为2049维
    """
    X_square = X ** 2  # 逐元素计算平方项
    X_poly2 = np.hstack((X, X_square))  # 拼接一次项和平方项
    return X_poly2

# -------------------------- 执行运行 --------------------------
if __name__ == "__main__":
    # 1. 加载数据
    train_X, train_y, val_X, val_y = load_data(DATA_ROOT, IMG_SIZE)
    # 2. 先训练一次项模型（基准对比）
    train_X1 = add_bias(train_X)
    val_X1 = add_bias(val_X)
    w1_base = linear_regression(train_X1, train_y, LAMBDA_BASE)
    val_y_pred1 = predict(val_X1, w1_base)
    acc_val1 = accuracy_score(val_y, val_y_pred1)
    # 3. 构建二次项特征并训练模型
    train_X2 = poly2_feature(train_X)
    val_X2 = poly2_feature(val_X)
    train_X2_design = add_bias(train_X2)  # 2049维设计矩阵
    val_X2_design = add_bias(val_X2)
    w2_base = linear_regression(train_X2_design, train_y, LAMBDA_BASE)
    # 4. 二次项模型预测与准确率
    train_y_pred2 = predict(train_X2_design, w2_base)
    val_y_pred2 = predict(val_X2_design, w2_base)
    acc_train2 = accuracy_score(train_y, train_y_pred2)
    acc_val2 = accuracy_score(val_y, val_y_pred2)
    # 5. 输出对比结果
    print("="*60)
    print("【环节3-多项式基函数扩展】执行完成")
    print(f"二次项特征维度（未加偏置）：{train_X2.shape}（样本数×2048维）")
    print(f"二次项设计矩阵维度（加偏置）：{train_X2_design.shape}（样本数×2049维）")
    print(f"二次项模型 - 训练集准确率：{acc_train2:.4f}，验证集准确率：{acc_val2:.4f}")
    print(f"一次项模型 - 验证集准确率：{acc_val1:.4f}")
    print(f"二次项相对一次项验证集准确率提升：{acc_val2 - acc_val1:.4f}")
    print("="*60)