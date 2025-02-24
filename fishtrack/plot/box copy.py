import os
import glob
import cv2
import numpy as np

# 数据集的路径
#image_dir = '/share/Lab_Datasets/fish_track/train/fish1/img1'
#label_dir = '/share/Lab_Datasets/fish_track/train/fish1/labels_with_ids'
#output_dir = '/share/Lab_Datasets/wd/track2/fish1'  # 输出的可视化数据集路径

image_dir = '/share/Lab_Datasets/fish_track/train/fish2/img1'
label_dir = '/share/Lab_Datasets/fish_track/train/fish2/labels_with_ids'
output_dir = '/share/Lab_Datasets/wd/track2/fish2'  # 输出的可视化数据集路径

#image_dir = '/share/Lab_Datasets/fish_track/train/fish4/img1'
#label_dir = '/share/Lab_Datasets/fish_track/train/fish4/labels_with_ids'
#output_dir = '/share/Lab_Datasets/wd/track2/fish4'  # 输出的可视化数据集路径

# 如果输出目录不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 遍历每个标签文件
for label_file in os.listdir(label_dir):
    label_file_path = os.path.join(label_dir, label_file)
    
    # 仅处理txt文件
    if not label_file.endswith('.txt'):
        continue
    
    # 读取标签文件内容
    with open(label_file_path, 'r') as f:
        lines = f.readlines()

    # 解析txt文件对应的图像文件名（自动匹配后缀）
    base_name = os.path.splitext(label_file)[0]  # 去掉 .txt 后缀
    image_candidates = glob.glob(os.path.join(image_dir, f"{base_name}.*"))  # 匹配所有扩展名
    if not image_candidates:
        print(f"Warning: 没有找到 {base_name} 的图像文件，跳过")
        continue
    image_path = image_candidates[0]  # 取第一个匹配的文件

    # 读取原始图像
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # 获取图像的宽度和高度

    # 遍历每一行，按 target_id 分类
    for line in lines:
        data = line.strip().split()
        if len(data) < 6:
            print(f"Warning: {label_file} 行数据格式异常: {data}")
            continue

        target_id = data[1].strip()  # 提取目标ID
        x_center = float(data[2]) * w  # x_center 转换为像素坐标
        y_center = float(data[3]) * h  # y_center 转换为像素坐标
        width = float(data[4]) * w  # width 转换为像素宽度
        height = float(data[5]) * h  # height 转换为像素高度

        # 计算目标框的左上角和右下角坐标
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 防止目标框超出图像边界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 绘制目标框（黄色）
        color = (0, 255, 255)  # 黄色 (BGR)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 设置目标ID的文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)  # 白色
        thickness = 2  # 稍微加粗字体
        text = str(target_id)

        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # 计算文本背景的坐标（比文本稍大）
        text_x1 = x2 - text_width - 10  # 右上角对齐，留出一点空隙
        text_y1 = y1 - text_height - 10
        text_x2 = x2
        text_y2 = y1

        # 确保背景不超出图像边界
        text_x1 = max(0, text_x1)
        text_y1 = max(0, text_y1)

        # 画出文本背景（黄色填充）
        cv2.rectangle(image, (text_x1, text_y1), (text_x2, text_y2), (0, 255, 255), -1)  # -1 代表填充颜色

        # 绘制目标ID文本
        cv2.putText(image, text, (text_x1 + 5, text_y2 - 5), font, font_scale, font_color, thickness)

    # 保存图像到指定目录
    output_image_path = os.path.join(output_dir, f"{base_name}_with_boxes.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"保存图像 {output_image_path}")

print("可视化构建完成！")
