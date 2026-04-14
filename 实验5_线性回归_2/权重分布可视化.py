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

# -------------------------- 依赖函数 --------------------------
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
    m = X.shape[0]
    bias = np.ones((m, 1))
    X_design = np.hstack((bias, X))
    return X_design


def linear_regression(X, y, lambda_):
    n = X.shape[1]
    X_T = X.T
    XTX = np.dot(X_T, X)
    lambda_I = lambda_ * np.eye(n)
    XTX_lambda = XTX + lambda_I
    XTX_inv = np.linalg.inv(XTX_lambda)
    w = np.dot(np.dot(XTX_inv, X_T), y)
    return w


# -------------------------- 核心函数：权重可视化 --------------------------
def weight_visualize(w, img_size, lambda_, save_name):
    w_no_bias = w[1:]
    w_img = w_no_bias.reshape(img_size, img_size)

    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(w_img, cmap="gray")
    plt.axis("off")
    # 图表标题保留英文（满足你的要求）
    plt.title(f"Weight Visualization (λ={lambda_:.2e})", fontsize=14, pad=20)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------- 执行运行 --------------------------
if __name__ == "__main__":
    train_X, train_y, val_X, val_y = load_data(DATA_ROOT, IMG_SIZE)
    train_X1 = add_bias(train_X)

    # 弱正则化
    lambda_weak = 1e-6
    w1_weak = linear_regression(train_X1, train_y, lambda_weak)
    weight_visualize(w1_weak, IMG_SIZE, lambda_weak, "weight_weak_lambda.png")

    # 强正则化
    lambda_strong = 1000.0
    w1_strong = linear_regression(train_X1, train_y, lambda_strong)
    weight_visualize(w1_strong, IMG_SIZE, lambda_strong, "weight_strong_lambda.png")

    # 控制台输出全部改为中文
    print("=" * 60)
    print("环节5 - 权重分布可视化 运行完成！")
    print(f"弱正则化（λ={lambda_weak:.2e}）权重图已保存：weight_weak_lambda.png")
    print(f"强正则化（λ={lambda_strong:.2e}）权重图已保存：weight_strong_lambda.png")
    print("=" * 60)