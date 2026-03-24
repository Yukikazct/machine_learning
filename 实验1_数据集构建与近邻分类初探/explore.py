# -*- coding: utf-8 -*-
"""
图像数据处理脚本：适配爬虫data路径
完成实验任务B+预处理A+预处理B
1. 任务B：打印单张图片Shape，提取高/宽/通道数
2. 预处理A：所有图片等比例缩放为64x64x3，保存至data/processed
3. 预处理B：为图片打标签（building=0，mountain=1），生成numpy格式数据集
运行方式：项目根目录执行 python code/data_process.py
"""
import os
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')  # 屏蔽图片处理的无关警告

# ===================== 1. 路径自动配置（和爬虫代码完全一致，无需修改）=====================
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')  # 爬虫原始数据根路径
# 预处理后图片保存路径：data/processed（自动创建，不覆盖原始数据）
PROCESSED_ROOT = os.path.join(DATA_ROOT, 'processed')
# 二分类类别（和爬虫CLASS_LIST一致，顺序对应标签0/1）
CLASS_LIST = ['building', 'mountain']
# 目标尺寸（实验要求64x64x3）
TARGET_SIZE = (64, 64)


# ===================== 2. 实验任务B：打印图片Shape，提取高/宽/通道数 =====================
def print_img_shape():
    """打印单张原始图片的Shape，完成实验任务B"""
    # 取第一张建筑类图片作为示例（也可改mountain）
    cls_dir = os.path.join(DATA_ROOT, CLASS_LIST[0])
    first_img = os.listdir(cls_dir)[0]
    first_img_path = os.path.join(cls_dir, first_img)

    # 读取图片并转为数组
    img = Image.open(first_img_path).convert('RGB')  # 强制转为RGB，避免灰度图（保证3通道）
    img_array = np.array(img)
    # 提取高度、宽度、通道数
    height, width, channel = img_array.shape

    # 打印结果（实验报告可直接复制）
    print("=" * 50)
    print("实验任务B：图片Shape与维度信息")
    print(f"示例图片路径：{first_img_path}")
    print(f"图片Shape：{img_array.shape}")
    print(f"图像高度：{height} 像素")
    print(f"图像宽度：{width} 像素")
    print(f"图像通道数：{channel} （RGB彩色图）")
    print("=" * 50 + "\n")
    return img


# ===================== 3. 预处理A：等比例缩放为64x64x3（核心：等比例，避免拉伸）=====================
def resize_img(img):
    """
    等比例缩放图片至64x64，强制RGB3通道
    :param img: PIL.Image对象
    :return: 缩放后的64x64x3 PIL.Image对象
    """
    img = img.convert('RGB')  # 统一转为RGB，解决灰度图/4通道透明图问题
    # 等比例缩放：先按短边缩放到目标尺寸，再居中裁剪（避免拉伸变形）
    img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)  # LANCZOS：高清缩放，适合实验
    # 创建64x64画布，居中粘贴缩放后的图片
    new_img = Image.new('RGB', TARGET_SIZE, (255, 255, 255))  # 白色背景填充空白
    paste_x = (TARGET_SIZE[0] - img.size[0]) // 2
    paste_y = (TARGET_SIZE[1] - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def batch_resize():
    """批量处理所有原始图片，缩放后保存至data/processed"""
    print("开始执行预处理A：批量缩放图片至64x64x3...\n")
    for cls in CLASS_LIST:
        # 创建每个类别对应的预处理文件夹
        cls_origin_dir = os.path.join(DATA_ROOT, cls)
        cls_processed_dir = os.path.join(PROCESSED_ROOT, cls)
        if not os.path.exists(cls_processed_dir):
            os.makedirs(cls_processed_dir)

        # 遍历该类别所有原始图片
        img_list = [f for f in os.listdir(cls_origin_dir) if f.endswith(('.jpg', '.png', '.bmp'))]
        for idx, img_name in enumerate(img_list, 1):
            img_path = os.path.join(cls_origin_dir, img_name)
            save_path = os.path.join(cls_processed_dir, f"{cls}_{idx}.jpg")  # 统一命名为jpg

            # 处理单张图片（跳过损坏图片）
            try:
                img = Image.open(img_path)
                resized_img = resize_img(img)
                resized_img.save(save_path, quality=95)  # 高质量保存
            except Exception as e:
                print(f"跳过损坏图片：{img_path}")
                continue

        print(f"【{cls}】缩放完成：{len(img_list)}张，保存至：{cls_processed_dir}")
    print("\n预处理A执行完成！所有图片已统一为64x64x3\n")


# ===================== 4. 预处理B：打标签，生成数据集（特征+标签）=====================
def create_dataset():
    """
    为预处理后的图片打标签，生成实验用数据集
    标签规则：building=0，mountain=1（可根据CLASS_LIST修改）
    返回：features(特征数组), labels(标签数组)，均为numpy格式（方便后续拆分/建模）
    """
    print("开始执行预处理B：为图片打标签，生成数据集...")
    features = []  # 特征：存储64x64x3的像素数组
    labels = []  # 标签：存储0/1

    for label, cls in enumerate(CLASS_LIST):
        cls_processed_dir = os.path.join(PROCESSED_ROOT, cls)
        img_list = [f for f in os.listdir(cls_processed_dir) if f.endswith('.jpg')]

        for img_name in img_list:
            img_path = os.path.join(cls_processed_dir, img_name)
            # 读取缩放后的图片，转为像素数组（64,64,3）
            img = Image.open(img_path)
            img_array = np.array(img)
            # 添加特征和标签
            features.append(img_array)
            labels.append(label)

    # 转为numpy数组（机器学习标准格式），特征归一化（像素值0-255→0-1，提升后续模型效果）
    features = np.array(features, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)
    # 保存数据集至data/processed（方便后续拆分/建模直接读取，无需重复处理）
    np.save(os.path.join(PROCESSED_ROOT, 'features.npy'), features)
    np.save(os.path.join(PROCESSED_ROOT, 'labels.npy'), labels)

    print(f"预处理B执行完成！")
    print(f"数据集规模：特征{features.shape} | 标签{labels.shape}")
    print(f"标签对应：{CLASS_LIST[0]}=0，{CLASS_LIST[1]}=1")
    print(f"数据集保存至：{PROCESSED_ROOT}/features.npy + labels.npy\n")
    return features, labels


# ===================== 主函数：按实验流程依次执行 =====================
if __name__ == '__main__':
    # 检查原始数据是否存在（避免路径错误）
    if not os.path.exists(DATA_ROOT) or not os.listdir(DATA_ROOT):
        raise FileNotFoundError("未找到原始数据！请先运行爬虫脚本crawler.py生成data文件夹")

    # 按实验流程执行
    print_img_shape()  # 任务B
    batch_resize()  # 预处理A
    create_dataset()  # 预处理B

    print("✅ 所有数据处理任务完成！下一步执行【实验4：数据集拆分】")