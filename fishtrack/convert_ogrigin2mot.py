import os
import cv2
import re

"""
将原始的 track_datasets 转换成 MOTChallenge 标准格式的标注文件

使用方法：
    1.修改如下三个绝对路径即可,变量名称不言自明
"""

# 设置绝对路径
images_dir = "/share/Lab_Datasets/track_dataset/fish4/images"
labels_dir = "/share/Lab_Datasets/track_dataset/fish4/labels_with_ids"
output_file = "/share/Lab_Datasets/track_dataset/fish4/gt/gt.txt"

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

# 获取图像名称
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg','.png'))],key=extract_number)

# 获取gt名称
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('txt')],key=extract_number)

# 确保数量一致
assert len(image_files) == len(label_files), "图片与标注数量不一致!"

# 处理每个帧的图像和标注
gt_entries = []

for img_file,label_file in zip(image_files,label_files):
    frame_id = extract_number(label_file)  # 解析帧号
    image_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, label_file)
    
    # 读取图片获取尺寸
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img_height, img_width = image.shape[:2]
    # 读取标注文件
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # 跳过格式错误的行

            _, obj_id, x_center, y_center, w, h = parts  # 第一个值忽略
            obj_id = int(obj_id)

            # 归一化坐标转换为整数像素坐标
            x_center, y_center, w, h = map(float, (x_center, y_center, w, h))
            tx = int((x_center - w / 2) * img_width)
            ty = int((y_center - h / 2) * img_height)
            w = int(w * img_width)
            h = int(h * img_height)

            # 添加到列表
            gt_entries.append((frame_id, obj_id, tx, ty, w, h))

# 按 ID 从小到大排序
gt_entries.sort(key=lambda x: x[1])

# 写入 gt.txt
with open(output_file, "w") as f:
    for entry in gt_entries:
        frame_id, obj_id, tx, ty, w, h = entry
        f.write(f"{frame_id}, {obj_id}, {tx}, {ty}, {w}, {h}, 0, 1, 1\n")

print(f"转换完成，已生成 {output_file}")