
"""
实验6：结果统计与归因分析（）
"""
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

plt.switch_backend('TkAgg')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 路径自动配置
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
PROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'data/processed')

EVAL_MODEL = 'sk'
# 类别定义：正类=1（山脉），负类=0（建筑）
POS_LABEL = 1
NEG_LABEL = 0
CLASS_NAMES = ['building(0)', 'mountain(1)']  # 可视化坐标轴标签

# ========== 路径调试（保留，定位文件问题）==========
print(f"【调试】脚本路径：{SCRIPT_PATH}")
print(f"【调试】npy文件路径：{PROCESSED_ROOT} | 路径是否存在：{os.path.exists(PROCESSED_ROOT)}")

def load_evaluation_data():
    """加载验证集真实标签+模型预测标签"""
    y_val_path = os.path.join(PROCESSED_ROOT, 'y_val.npy')
    y_pred_path = os.path.join(PROCESSED_ROOT, f'y_val_pred_{EVAL_MODEL}.npy')
    # 检查文件是否存在
    print(f"\n【调试】真实标签：{y_val_path} | 存在：{os.path.exists(y_val_path)}")
    print(f"【调试】预测标签：{y_pred_path} | 存在：{os.path.exists(y_pred_path)}")
    if not os.path.exists(y_val_path):
        raise FileNotFoundError(f"未找到y_val.npy！请先运行实验4数据集拆分")
    if not os.path.exists(y_pred_path):
        raise FileNotFoundError(f"未找到{y_pred_path}！请先运行实验5的1-NN模型")
    # 加载数据
    y_val = np.load(y_val_path)
    y_val_pred = np.load(y_pred_path)
    model_name = 'sklearn版' if EVAL_MODEL == 'sk' else '手动编写版'
    print(f"\n 加载成功 | 评估模型：{model_name} | 验证集样本数：{len(y_val)}")
    return y_val, y_val_pred, model_name

def cal_confusion_matrix_manual(y_true, y_pred):
    """自行编写：计算混淆矩阵+TP/FP/FN/TN（实验核心要求）"""
    cm = np.zeros((2, 2), dtype=int)
    TP = np.sum((y_true == POS_LABEL) & (y_pred == POS_LABEL))
    FP = np.sum((y_true == NEG_LABEL) & (y_pred == POS_LABEL))
    FN = np.sum((y_true == POS_LABEL) & (y_pred == NEG_LABEL))
    TN = np.sum((y_true == NEG_LABEL) & (y_pred == NEG_LABEL))
    cm[0, 0] = TN
    cm[0, 1] = FP
    cm[1, 0] = FN
    cm[1, 1] = TP
    return cm, TP, FP, FN, TN

# ========== 核心新增：可视化混淆矩阵（窗口显示）==========
def plot_confusion_matrix(cm, class_names, model_name):
    """
    绘制混淆矩阵热力图，弹出独立窗口显示
    :param cm: 手动计算的混淆矩阵
    :param class_names: 类别名称（x/y轴标签）
    :param model_name: 模型名称（标题）
    """
    fig, ax = plt.subplots(figsize=(6, 5))  # 设置窗口大小
    # 绘制热力图，蓝色系更直观
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)  # 颜色条
    # 设置坐标轴标签、标题
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title=f'1-NN Confusion Matrix - {model_name}\n(building=0 / mountain=1)')
    # 旋转x轴标签，防止重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # 在热力图上标注数值（核心：显示TP/FP/FN/TN具体值）
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # 紧凑布局，防止内容裁剪
    fig.tight_layout()
    # 显示窗口（阻塞式，关闭窗口后代码继续执行误差分析）
    plt.show()

def model_evaluation(y_true, y_pred, model_name):
    """实验6任务A：终端打印+窗口可视化混淆矩阵"""
    cm, TP, FP, FN, TN = cal_confusion_matrix_manual(y_true, y_pred)
    # 终端打印结果（实验报告可复制）
    total = TP + FP + FN + TN
    acc = (TP + TN) / total if total != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    print(f"\n===== 实验6任务A：{model_name}1-NN 混淆矩阵统计 =====")
    print(f"混淆矩阵（2x2）：[[TN, FP],[FN, TP]]\n{cm}")
    print(f"核心指标：TP={TP} | FP={FP} | FN={FN} | TN={TN}")
    print(f"评估指标：准确率={acc:.4f} | 精确率={precision:.4f} | 召回率={recall:.4f}")
    # 新增：调用可视化函数，弹出窗口显示混淆矩阵
    plot_confusion_matrix(cm, CLASS_NAMES, model_name)
    return cm, TP, FP, FN, TN

def error_analysis_manual(y_true, y_pred):
    """实验6任务B：误差分析（像素级比较的影响）"""
    error_idx = np.where(y_true != y_pred)[0]
    error_num = len(error_idx)
    error_rate = error_num / len(y_true) if len(y_true) != 0 else 0
    print(f"\n===== 实验6任务B：分类错误分析与归因 =====")
    print(f"验证集总样本：{len(y_true)} | 错误样本：{error_num} | 错误率：{error_rate:.4f}")
    if error_num > 0:
        print(f"错误样本索引：{error_idx.tolist()[:10]}")  # 显示前10个
        print(f"错误标签示例：{[(y_true[i], y_pred[i]) for i in error_idx[:5]]}")
    # 误差归因（贴合实验要求：像素级比较的影响）
    print(f"\n📌 像素级比较（1-NN）分类错误的核心原因：")
    reasons = [
        "1. 背景干扰：目标物体占比小，天空/岩石等背景像素主导匹配",
        "2. 像素相似：建筑岩石墙面/玻璃反光与山脉像素分布高度相似",
        "3. 缩放失真：64x64缩放导致目标细节像素丢失，特征模糊",
        "4. 光照角度：同类物体不同光照/角度，像素分布差异大",
        "5. 算法局限：仅关注像素值差异，未提取图像语义特征",
        "6. 样本质量：爬取图片存在模糊/噪点，像素值失真导致距离计算偏差"
    ]
    for idx, reason in enumerate(reasons, 1):
        print(f"   {reason}")
    print(f"\n✅ 实验6完成！混淆矩阵窗口已显示，终端结果可复制至实验报告")

if __name__ == '__main__':
    # 全局异常捕获，显示具体错误
    try:
        y_val, y_val_pred, model_name = load_evaluation_data()
        model_evaluation(y_val, y_val_pred, model_name)  # 终端打印+窗口可视化
        error_analysis_manual(y_val, y_val_pred)
    except Exception as e:
        print(f"\n❌ 执行错误：{type(e).__name__} - {str(e)}")
        print("👉 解决：1.检查data/processed下是否有y_val.npy和预测npy 2.确认路径正确")