# -*- coding: utf-8 -*-
"""
图片爬虫脚本：爬取指定二分类图片各50张，自动存入项目根目录的data文件夹
运行方式：任意目录执行 python code/crawler.py 均可
可修改：CLASS_LIST 替换为自己的二分类类别（如['building', 'forest']）
"""
import requests
from bs4 import BeautifulSoup
import os
import time

# ===================== 1. 基础配置（可根据需求修改）=====================
HEADERS = {  # 模拟浏览器，避免反爬
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
}
CLASS_LIST = ['building', 'mountain']  # 二分类类别，可替换为['建筑', '森林']等
IMG_NUM = 50  # 每个类别爬取50张（匹配实验要求）
BASE_URL = 'https://cn.bing.com/images/search?q={}&first={}'  # 必应图片（反爬弱，适合新手）

# ===================== 2. 自动计算路径（核心：无需手动修改）=====================
# 获取当前爬虫脚本（code.py）的绝对路径
SCRIPT_PATH = os.path.abspath(__file__)
# 获取项目根目录（code的上级目录，即包含code和data的文件夹）
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
# 图片最终存储的根路径：项目根目录/data
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')


# ===================== 3. 图片下载核心函数 =====================
def download_img(class_name, save_dir):
    """
    下载指定类别的图片
    :param class_name: 类别名（如cat）
    :param save_dir: 该类别图片的保存路径（如data/cat）
    """
    img_count = 0  # 已下载有效图片计数
    page = 0  # 分页索引（必应图片每页35张）
    print(f'========== 开始爬取【{class_name}】，目标{IMG_NUM}张，保存路径：{save_dir} ==========')

    while img_count < IMG_NUM:
        # 构造分页搜索链接
        search_url = BASE_URL.format(class_name, page)
        try:
            # 请求搜索页面
            res = requests.get(search_url, headers=HEADERS, timeout=10)
            res.raise_for_status()  # 抛出请求错误（如403/500）
            soup = BeautifulSoup(res.text, 'html.parser')
            # 解析所有图片标签（必应图片核心标签）
            img_tags = soup.find_all('img', class_='mimg')

            for img in img_tags:
                if img_count >= IMG_NUM:  # 达到50张则停止
                    break
                # 获取图片真实链接（src优先，无则取data-src）
                img_url = img.get('src') or img.get('data-src')
                if not img_url:  # 无链接则跳过
                    continue

                try:
                    # 下载图片
                    img_res = requests.get(img_url, headers=HEADERS, timeout=5)
                    img_res.raise_for_status()
                    # 图片命名：类别_序号.jpg（如cat_1.jpg）
                    img_name = f'{class_name}_{img_count + 1}.jpg'
                    img_path = os.path.join(save_dir, img_name)
                    # 保存图片
                    with open(img_path, 'wb') as f:
                        f.write(img_res.content)

                    img_count += 1
                    print(f'已下载：{img_name}（当前{img_count}/{IMG_NUM}）')
                    time.sleep(0.5)  # 延迟0.5秒，避免请求过快被反爬

                except Exception as e:
                    # 单张图片下载失败（如链接失效），直接跳过
                    continue

            page += 35  # 翻页（必应图片分页步长35）

        except Exception as e:
            # 页面请求失败，直接翻页
            page += 35
            continue

    print(f'========== 【{class_name}】爬取完成，实际下载{img_count}张 ==========\n')


# ===================== 4. 主函数：遍历类别爬取 =====================
if __name__ == '__main__':
    # 遍历每个类别，创建文件夹并下载
    for cls in CLASS_LIST:
        # 该类别图片的保存路径（如data/cat）
        cls_save_dir = os.path.join(DATA_ROOT, cls)
        # 若文件夹不存在则自动创建
        if not os.path.exists(cls_save_dir):
            os.makedirs(cls_save_dir)
        # 下载该类别图片
        download_img(cls, cls_save_dir)

    print('========== 所有类别爬取完成！==========')
    print(f'温馨提示：请前往 {DATA_ROOT} 检查图片，手动剔除无关/损坏的，不足50张请重新运行脚本补充')