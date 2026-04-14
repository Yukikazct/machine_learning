import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# -------------------------- 全局配置 --------------------------
DATA_ROOT = "data"  # 数据集根路径
IMG_SIZE = 32        # 图像缩放为32x32
LAMBDA_BASE = 1e-6   # 极小正则化系数
LAMBDA_LIST = [1e-6, 1e-3, 1.0, 10.0, 100.0, 1000.0]

# -------------------------- 核心函数 --------------------------
def load_data(data_root, img_size):

    train_X, train_y = [], []
    val_X, val_y = [], []
    # 遍历训练集/验证集，猫/狗类别
    for phase in ["train", "val"]:
        for label, cls in enumerate(["cat", "dog"]):
            cls_path = os.path.join(data_root, phase, cls)
            # 遍历类别下所有图片
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue  # 跳过损坏图片
                img_resized = cv2.resize(img, (img_size, img_size))  # 32x32
                img_norm = img_resized / 255.0  # 归一化
                img_flat = img_norm.flatten()   # 展平为1024维
                # 分配到对应数据集
                if phase == "train":
                    train_X.append(img_flat)
                    train_y.append(label)
                else:
                    val_X.append(img_flat)
                    val_y.append(label)
    # 转为numpy数组（机器学习标准格式）
    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32).reshape(-1, 1)
    val_X = np.array(val_X, dtype=np.float32)
    val_y = np.array(val_y, dtype=np.float32).reshape(-1, 1)
    return train_X, train_y, val_X, val_y

# -------------------------- 执行运行 --------------------------
if __name__ == "__main__":
    train_X, train_y, val_X, val_y = load_data(DATA_ROOT, IMG_SIZE)
    print("="*60)
    print("【环节1-图像数据标准化】执行完成")
    print(f"训练集特征维度：{train_X.shape}（样本数×1024维）")
    print(f"验证集特征维度：{val_X.shape}（样本数×1024维）")
    print(f"标签规则：猫=0，狗=1，训练集标签维度：{train_y.shape}")
    print("="*60)