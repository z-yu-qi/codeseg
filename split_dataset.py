import os
import random
import shutil
from pathlib import Path

def split_dataset(train_dir, valid_dir, split_ratio=0.2):
    """
    将训练集数据随机划分一部分到验证集
    
    Args:
        train_dir: 训练集目录
        valid_dir: 验证集目录
        split_ratio: 验证集比例，默认0.2（20%）
    """
    # 确保目录存在
    train_images = Path(train_dir) / 'images'
    train_labels = Path(train_dir) / 'labels'
    valid_images = Path(valid_dir) / 'images'
    valid_labels = Path(valid_dir) / 'labels'
    
    for dir_path in [valid_images, valid_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_files = list(train_images.glob('*.jpg')) + list(train_images.glob('*.png'))
    total_images = len(image_files)
    num_valid = int(total_images * split_ratio)
    
    print(f"总图片数量: {total_images}")
    print(f"验证集数量: {num_valid}")
    
    # 随机选择验证集图片
    valid_images_selected = random.sample(image_files, num_valid)
    
    # 移动文件
    for img_path in valid_images_selected:
        # 获取对应的标签文件路径
        label_path = train_labels / (img_path.stem + '.txt')
        
        # 目标路径
        valid_img_path = valid_images / img_path.name
        valid_label_path = valid_labels / (img_path.stem + '.txt')
        
        # 移动图片和标签
        if label_path.exists():
            print(f"移动: {img_path.name}")
            shutil.move(str(img_path), str(valid_img_path))
            shutil.move(str(label_path), str(valid_label_path))
        else:
            print(f"警告: 找不到标签文件 {label_path}")

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    random.seed(42)
    
    # 设置目录路径
    train_dir = "datasets/data/train"
    valid_dir = "datasets/data/valid"
    
    # 执行划分
    split_dataset(train_dir, valid_dir) 