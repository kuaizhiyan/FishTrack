import os
import cv2
import re

# 设置路径
images_dir = "/share/Lab_Datasets/fish_track/train/fish1/img1"
labels_dir = "/share/Lab_Datasets/fish_track/train/fish1/labels_with_ids"
output_file = "/share/Lab_Datasets/fish_track/train/fish1/det/det.txt"

# 提取文件名中的数字部分，用于排序
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

# 获取按数字排序的图片文件
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))], key=extract_number)

# 获取按数字排序的标注文件
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')], key=extract_number)

# 确保文件数量一致
if len(image_files) != len(label_files):
    raise ValueError("图片数量与标注文件数量不匹配，请检查数据集")

# 存储检测结果
det_entries = []

# 遍历所有帧
for img_file, label_file in zip(image_files, label_files):
    frame_id = extract_number(label_file)  # 获取帧号
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

            _, obj_id, x_center, y_center, w, h = parts  # 忽略 class_id
            obj_id = int(obj_id)

            # 归一化坐标转换为浮点数像素坐标，并保留 1 位小数
            x_center, y_center, w, h = map(float, (x_center, y_center, w, h))
            tx = round((x_center - w / 2) * img_width, 1)
            ty = round((y_center - h / 2) * img_height, 1)
            w = round(w * img_width, 1)
            h = round(h * img_height, 1)

            # 置信度 conf 统一设为 1.0
            det_entries.append(f"{frame_id}, -1 , {tx:.1f}, {ty:.1f}, {w:.1f}, {h:.1f}, 1.0, -1, -1, -1")

# 按 frame_id 排序
det_entries.sort(key=lambda x: int(x.split(",")[0]))

# 写入 det.txt
with open(output_file, "w") as f:
    f.write("\n".join(det_entries))

print(f"转换完成，已生成 {output_file}")
