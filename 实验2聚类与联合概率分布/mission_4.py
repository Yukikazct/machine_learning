import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import warnings
from matplotlib import rcParams

warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'sans-serif']
rcParams['axes.unicode_minus'] = False

def kmeans(data, K, max_iter=100, tol=1e-6):
    N, D = data.shape
    centers = data[np.random.choice(N, K, replace=False)]
    
    for _ in range(max_iter):
        dists = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(dists, axis=0)
        
        new_centers = np.zeros((K, D))
        for k in range(K):
            if np.sum(labels == k) > 0:
                new_centers[k] = data[labels == k].mean(axis=0)
            else:
                new_centers[k] = data[np.random.choice(N, 1)]
        
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    
    return labels, centers

def extract_image_feature(img_path, target_size=(64, 64)):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        return img_array.flatten()
    except Exception as e:
        print(f"读取图片 {img_path} 失败：{e}")
        return None

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = script_dir

label_mapping = {"mountain": 0, "building": 1}

features = []
labels_true = []
img_paths = []

for class_name in ["mountain", "building"]:
    class_dir = os.path.join(root_dir, class_name)
    if not os.path.exists(class_dir):
        print(f"  警告：文件夹 {class_dir} 不存在，请检查路径！")
        continue

    for filename in os.listdir(class_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, filename)
            feat = extract_image_feature(img_path)
            if feat is not None:
                features.append(feat)
                labels_true.append(label_mapping[class_name])
                img_paths.append(img_path)

X = np.array(features)
y_true = np.array(labels_true)

if len(X) == 0:
    print(" 错误：没有加载到任何图片！请检查图片和路径。")
    exit()

print(f" 数据加载完成！共读取 {X.shape[0]} 张图片，特征维度 {X.shape[1]}")
print(f"类别分布：山 (0): {np.sum(y_true==0)} 张，建筑 (1): {np.sum(y_true==1)} 张")

K = 2

if len(X) < K:
    print(f" 错误：样本数量 ({len(X)}) 小于聚类数 ({K})")
    exit()

labels_pred, centers = kmeans(X, K)

acc1 = np.mean(labels_pred == y_true)
acc2 = np.mean(labels_pred != y_true)
accuracy = max(acc1, acc2)

is_reversed = acc2 > acc1
if is_reversed:
    labels_pred_aligned = 1 - labels_pred
else:
    labels_pred_aligned = labels_pred

correct_mask = (labels_pred_aligned == y_true)
incorrect_mask = ~correct_mask

n_correct = np.sum(correct_mask)
n_incorrect = np.sum(incorrect_mask)

print(f"\n🎯 任务 A & B：聚类结果统计")
print(f"=" * 50)
print(f"总样本数：{len(y_true)} 张")
print(f"聚类正确：{n_correct} 张 ({n_correct/len(y_true)*100:.2f}%)")
print(f"聚类错误：{n_incorrect} 张 ({n_incorrect/len(y_true)*100:.2f}%)")
print(f"聚类准确率：{accuracy:.2%}")

if n_incorrect > 0:
    print(f"\n 错误样本分析:")
    incorrect_indices = np.where(incorrect_mask)[0]
    for idx in incorrect_indices[:min(5, n_incorrect)]:
        true_label = ['山', '建筑'][y_true[idx]]
        pred_label = ['山', '建筑'][labels_pred_aligned[idx]]
        print(f"  - 图片 {os.path.basename(img_paths[idx])}: 真实={true_label}, 预测={pred_label}")

print(f"\n 任务 C：可视化聚类中心")
print(f"=" * 50)

plt.figure(figsize=(15, 5))

for i in range(K):
    cluster_size = np.sum(labels_pred == i)
    print(f"聚类中心 {i}: 包含 {cluster_size} 个样本")
    
    plt.subplot(1, 3, i+1)
    center_img = centers[i].reshape(64, 64, 3)
    center_img_normalized = (center_img - center_img.min()) / (center_img.max() - center_img.min() + 1e-8)
    plt.imshow(center_img_normalized)
    plt.title(f"聚类中心 {i}\n(属于该类：{cluster_size}张)")
    plt.axis('off')

plt.tight_layout()
plt.show()

if n_incorrect > 0:
    print(f"\n 错误样本可视化分析:")
    n_show = min(n_incorrect, 10)
    incorrect_indices = np.where(incorrect_mask)[0][:n_show]
    
    n_rows = (n_show + 4) // 5
    plt.figure(figsize=(20, 4*n_rows))
    
    for idx_pos, idx in enumerate(incorrect_indices):
        plt.subplot(n_rows, 5, idx_pos+1)
        img = Image.open(img_paths[idx])
        img_array = np.array(img)
        plt.imshow(img_array)
        
        true_label = ['山', '建筑'][y_true[idx]]
        pred_label = ['山', '建筑'][labels_pred_aligned[idx]]
        
        plt.title(f" 分类错误\n真实：{true_label}\n预测：{pred_label}",
                 fontsize=10, color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

print(f"\n 正确与错误样本对比展示:")

n_correct_show = min(5, n_correct)
n_incorrect_show = min(5, n_incorrect)
total_show = n_correct_show + n_incorrect_show

if total_show > 0:
    plt.figure(figsize=(20, 4))
    
    correct_indices = np.where(correct_mask)[0][:n_correct_show]
    
    for idx_pos, idx in enumerate(correct_indices):
        plt.subplot(2, 5, idx_pos+1)
        img = Image.open(img_paths[idx])
        plt.imshow(img)
        true_label = ['山', '建筑'][y_true[idx]]
        plt.title(f" 正确\n真实：{true_label}\n预测：{true_label}",
                 fontsize=9, color='green')
        plt.axis('off')
    
    incorrect_indices = np.where(incorrect_mask)[0][:n_incorrect_show]
    
    for idx_pos, idx in enumerate(incorrect_indices):
        plt.subplot(2, 5, n_correct_show + idx_pos + 1)
        img = Image.open(img_paths[idx])
        plt.imshow(img)
        true_label = ['山', '建筑'][y_true[idx]]
        pred_label = ['山', '建筑'][labels_pred_aligned[idx]]
        plt.title(f" 错误\n真实：{true_label}\n预测：{pred_label}",
                 fontsize=9, color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

print(f"\n 聚类结果分析总结:")
print(f"=" * 50)
print(f"1. 数据集包含 {len(y_true)} 张图像，分为 2 类（山和建筑）")
print(f"2. 使用 K-Means 算法进行无监督聚类 (K=2)")
print(f"3. 最终聚类准确率为：{accuracy:.2%}")
if accuracy >= 0.8:
    print(f"4. 评价：聚类效果良好！✅")
elif accuracy >= 0.6:
    print(f"4. 评价：聚类效果中等，有一定分类错误 ️")
else:
    print(f"4. 评价：聚类效果较差，建议优化特征提取 ")

if n_incorrect > 0:
    print(f"\n 可能的错误原因分析:")
    print(f"  - 某些山景和建筑景观的视觉特征相似（如山中包含建筑物）")
    print(f"  - 图像颜色、纹理特征不够区分明显")
    print(f"  - K-Means 对初始值敏感，可能陷入局部最优")
    print(f"  - 建议使用更复杂的特征（如 SIFT、CNN 特征）提升效果")
