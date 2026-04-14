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


def predict(X_design, w, threshold=0.5):
    y_pred = np.dot(X_design, w)
    y_pred_label = (y_pred > threshold).astype(np.float32)
    return y_pred_label


def poly2_feature(X):
    X_square = X ** 2
    X_poly2 = np.hstack((X, X_square))
    return X_poly2


# -------------------------- 核心函数 --------------------------
def reg_sweep(X, y, X_val, y_val, lambda_list, is_poly2=False):
    acc_train_list = []
    acc_val_list = []
    if is_poly2:
        X = poly2_feature(X)
        X_val = poly2_feature(X_val)
    X_design = add_bias(X)
    X_val_design = add_bias(X_val)

    for lam in lambda_list:
        w = linear_regression(X_design, y, lam)
        y_pred_train = predict(X_design, w)
        y_pred_val = predict(X_val_design, w)
        acc_train = accuracy_score(y, y_pred_train)
        acc_val = accuracy_score(y_val, y_pred_val)
        acc_train_list.append(acc_train)
        acc_val_list.append(acc_val)
        # 控制台打印改为中文
        print(f"λ={lam:.2e} → 训练准确率：{acc_train:.4f}，验证准确率：{acc_val:.4f}")
    return acc_train_list, acc_val_list


# -------------------------- 执行运行 --------------------------
if __name__ == "__main__":
    train_X, train_y, val_X, val_y = load_data(DATA_ROOT, IMG_SIZE)
    print("=" * 40)
    # 改为中文
    print("一次项模型 - 正则化λ扫描结果：")
    acc_train1_sweep, acc_val1_sweep = reg_sweep(train_X, train_y, val_X, val_y, LAMBDA_LIST, is_poly2=False)
    print("=" * 40)
    # 改为中文
    print("二次项模型 - 正则化λ扫描结果：")
    acc_train2_sweep, acc_val2_sweep = reg_sweep(train_X, train_y, val_X, val_y, LAMBDA_LIST, is_poly2=True)

    # 图表全英文（保持不变）
    plt.figure(figsize=(12, 5), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(LAMBDA_LIST, acc_train1_sweep, marker='o', color='#1f77b4', label="Train Set")
    plt.plot(LAMBDA_LIST, acc_val1_sweep, marker='s', color='#ff7f0e', label="Validation Set")
    plt.xscale("log")
    plt.xlabel("Regularization λ (log scale)")
    plt.ylabel("Accuracy")
    plt.title("1st-order Model: λ-Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(LAMBDA_LIST, acc_train2_sweep, marker='o', color='#1f77b4', label="Train Set")
    plt.plot(LAMBDA_LIST, acc_val2_sweep, marker='s', color='#ff7f0e', label="Validation Set")
    plt.xscale("log")
    plt.xlabel("Regularization λ (log scale)")
    plt.ylabel("Accuracy")
    plt.title("2nd-order Model: λ-Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lambda_accuracy_curve.png", dpi=300, bbox_inches='tight')
    print("=" * 60)
    # 改为中文
    print("环节4 - 正则化强度扫描 运行完成！")
    print("λ-准确率曲线已保存为：lambda_accuracy_curve.png")
    print("=" * 60)
    plt.show()