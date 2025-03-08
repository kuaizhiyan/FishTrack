import os
import json
from tqdm import tqdm
from PIL import Image

def convert_fishtrack_to_coco(root_folder, output_json):
    """
    将 FishTrack 数据集转换为 COCO 格式
    :param root_folder: FishTrack 数据集的根路径
    :param output_json: 输出的 COCO 格式 JSON 文件路径
    """
    # COCO 格式的基本结构
    coco_format = {
        "info": {
            "description": "FishTrack Dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": "2023-10-01"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "fish",
                "supercategory": "animal"
            }
        ]
    }

    # 初始化变量
    image_id = 1
    annotation_id = 1

    # 获取 train 文件夹路径
    train_folder = os.path.join(root_folder, "train")
    if not os.path.exists(train_folder):
        raise ValueError(f"Train folder not found in {root_folder}")

    # 遍历 train 文件夹下的所有子文件夹（如 fish1, fish2, fish4）
    for sequence in os.listdir(train_folder):
        sequence_path = os.path.join(train_folder, sequence)
        if not os.path.isdir(sequence_path):
            continue

        # 检查是否存在 img1 和 labels_with_ids 文件夹
        img_folder = os.path.join(sequence_path, "img1")
        label_folder = os.path.join(sequence_path, "labels_with_ids")
        if not (os.path.exists(img_folder) and os.path.exists(label_folder)):
            continue

        # 遍历图像文件夹
        for img_name in tqdm(os.listdir(img_folder), desc=f"Processing {sequence}"):
            if not img_name.endswith(".jpg"):
                continue

            # 读取图像尺寸
            img_path = os.path.join(img_folder, img_name)
            with Image.open(img_path) as img:
                width, height = img.size

            # 添加图像信息到 COCO 格式
            coco_format["images"].append({
                "id": image_id,
                "file_name": os.path.join("train", sequence, "img1", img_name),  # 相对于根路径
                "width": width,
                "height": height
            })

            # 读取对应的标注文件
            label_path = os.path.join(label_folder, img_name.replace(".jpg", ".txt"))
            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                lines = f.readlines()

            # 解析标注文件
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue

                # 解析目标框信息
                track_id = int(parts[1])  # 轨迹 ID（忽略）
                x_center = float(parts[2])  # 中心点 x 坐标（归一化）
                y_center = float(parts[3])  # 中心点 y 坐标（归一化）
                bbox_width = float(parts[4])  # 宽度（归一化）
                bbox_height = float(parts[5])  # 高度（归一化）

                # 将归一化坐标转换为绝对坐标
                x = (x_center - bbox_width / 2) * width
                y = (y_center - bbox_height / 2) * height
                w = bbox_width * width
                h = bbox_height * height

                # 添加标注信息到 COCO 格式
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # 类别 ID（鱼类）
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })

                annotation_id += 1

            image_id += 1

    # 创建 annotations 文件夹（如果不存在）
    annotations_folder = os.path.join(root_folder, "annotations")
    os.makedirs(annotations_folder, exist_ok=True)

    # 保存为 JSON 文件
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"转换完成！COCO 格式文件已保存到 {output_json}")


# 使用示例
root_folder = "/share/Lab_Datasets/fish_track"  # FishTrack 数据集的根路径
output_json = os.path.join(root_folder, "annotations", "fishtrack_coco.json")  # 输出的 COCO 格式 JSON 文件路径
convert_fishtrack_to_coco(root_folder, output_json)