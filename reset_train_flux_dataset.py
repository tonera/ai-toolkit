# 此脚本将重置flux kontext的lora训练数据集和环境

# 功能:将数据集目录下所有的txt文件添加或者删除指定的标签。如果没有txt文件，则根据图片文件名生成txt文件。

import os
import argparse
import glob
from pathlib import Path


def get_image_files(dataset_dir):
    """获取数据集目录下的所有图片文件"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_dir, ext)))
        image_files.extend(glob.glob(os.path.join(dataset_dir, ext.upper())))
    
    return sorted(image_files)


def get_txt_files(dataset_dir):
    """获取数据集目录下的所有txt文件"""
    txt_files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    return sorted(txt_files)


def generate_txt_from_image(image_path, default_tags=None):
    """根据图片文件名生成txt文件内容"""
    if default_tags is None:
        default_tags = []
    
    # 只使用默认标签，不从文件名生成标签
    all_tags = []
    for tag in default_tags:
        if tag not in all_tags:  # 避免重复添加
            all_tags.append(tag)
    
    # 返回（使用逗号分隔）
    return ', '.join(all_tags)


def process_txt_file(txt_path, action, tags, default_tags=None, allow_empty=False):
    """处理单个txt文件，添加或删除标签"""
    if not os.path.exists(txt_path):
        return
    
    # 读取现有内容
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 处理逗号分隔的标签
    if content:
        # 先按逗号分割，然后去除每个标签的前后空格
        existing_tags = [tag.strip() for tag in content.split(',')]
        # 过滤掉空标签
        existing_tags = [tag for tag in existing_tags if tag]
    else:
        existing_tags = []
    
    if action == 'add':
        # 添加标签到末尾，保持原始顺序
        new_tags = existing_tags.copy()
        for tag in tags:
            if tag not in new_tags:  # 避免重复添加
                new_tags.append(tag)
    elif action == 'remove':
        # 删除标签
        new_tags = [tag for tag in existing_tags if tag not in tags]
        # 如果删除后没有标签了，根据设置决定是否保留原内容
        if not new_tags and existing_tags and not allow_empty:
            print(f"警告: 删除标签后文件将为空，保留原内容: {txt_path}")
            return
    else:
        print(f"未知操作: {action}")
        return
    
    # 写入新内容（保持逗号分隔格式）
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(', '.join(new_tags))
    
    print(f"已处理: {txt_path}")


def create_txt_for_image(image_path, default_tags=None):
    """为图片创建对应的txt文件"""
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    
    if os.path.exists(txt_path):
        print(f"txt文件已存在，跳过: {txt_path}")
        return
    
    content = generate_txt_from_image(image_path, default_tags)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已创建: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description='处理flux训练数据集的标签文件')
    parser.add_argument('dataset_dir', help='数据集目录路径')
    parser.add_argument('--action', choices=['add', 'remove'], default='add', 
                       help='操作类型: add(添加标签) 或 remove(删除标签)')
    parser.add_argument('--tags', nargs='*', help='要添加或删除的标签列表（用空格分隔多个标签）')
    parser.add_argument('--tag-string', help='要添加或删除的完整标签字符串（包含空格的长标签）')
    parser.add_argument('--default-tags', nargs='+', default=[], 
                       help='为没有txt文件的图片添加的默认标签')
    parser.add_argument('--create-missing', action='store_true', 
                       help='为没有txt文件的图片创建txt文件')
    parser.add_argument('--allow-empty', action='store_true', 
                       help='允许删除标签后文件为空（默认会保留原内容）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"错误: 数据集目录不存在: {args.dataset_dir}")
        return
    
    # 处理标签参数
    tags_to_process = []
    if args.tags:
        tags_to_process.extend(args.tags)
    if args.tag_string:
        tags_to_process.append(args.tag_string)
    
    print(f"处理数据集目录: {args.dataset_dir}")
    print(f"操作类型: {args.action}")
    
    if tags_to_process:
        print(f"标签: {tags_to_process}")
    
    # 获取所有txt文件
    txt_files = get_txt_files(args.dataset_dir)
    print(f"找到 {len(txt_files)} 个txt文件")
    
    # 处理现有的txt文件
    if tags_to_process and txt_files:
        for txt_file in txt_files:
            process_txt_file(txt_file, args.action, tags_to_process, args.default_tags, args.allow_empty)
    
    # 如果需要创建缺失的txt文件
    if args.create_missing:
        image_files = get_image_files(args.dataset_dir)
        print(f"找到 {len(image_files)} 个图片文件")
        
        for image_file in image_files:
            txt_file = os.path.splitext(image_file)[0] + '.txt'
            if not os.path.exists(txt_file):
                # 创建txt文件时，将default_tags和tags_to_process合并
                all_default_tags = list(args.default_tags)
                if tags_to_process:
                    all_default_tags.extend(tags_to_process)
                create_txt_for_image(image_file, all_default_tags)
    
    print("处理完成!")


if __name__ == "__main__":
    main()
