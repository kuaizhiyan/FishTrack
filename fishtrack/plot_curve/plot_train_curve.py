import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.ndimage import gaussian_filter1d

def read_json_log(file_path, metric="accuracy_top-1"):
    """读取 JSON 训练日志，并返回 step 和 metric 数据"""
    steps, values = [], []

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())  # 解析 JSON
            if metric in data:
                steps.append(data["step"])
                values.append(data[metric])

    return np.array(steps), np.array(values)

def smooth_curve(x, y, method="moving_avg", window_size=10):
    """
    平滑曲线
    :param x: 原始 step 值
    :param y: 原始指标值
    :param method: 选择的平滑方法（moving_avg, ema, gaussian, savgol, peaks）
    :param window_size: 滑动窗口大小
    :return: 平滑后的 x, y
    """
    if len(y) < window_size:  # 避免窗口大小过大
        return x, y

    if method == "moving_avg":  # 滑动平均平滑
        y_smooth = np.convolve(y, np.ones(window_size) / window_size, mode="valid")
        x_smooth = x[:len(y_smooth)]  # 保持一致长度

    elif method == "ema":  # 指数加权移动平均 (EMA)
        alpha = 2 / (window_size + 1)
        y_smooth = np.zeros_like(y)
        y_smooth[0] = y[0]  # 初始值
        for i in range(1, len(y)):
            y_smooth[i] = alpha * y[i] + (1 - alpha) * y_smooth[i - 1]
        x_smooth = x

    elif method == "gaussian":  # 高斯滤波平滑
        y_smooth = gaussian_filter1d(y, sigma=window_size / 3)
        x_smooth = x

    elif method == "savgol":  # Savitzky-Golay 滤波
        y_smooth = savgol_filter(y, window_length=window_size, polyorder=2)
        x_smooth = x

    elif method == "peaks":  # 只选择波峰
        peaks = argrelextrema(y, np.greater)[0]
        sampled_indices = np.linspace(0, len(peaks) - 1, min(len(peaks), 2000), dtype=int)
        x_smooth, y_smooth = x[peaks][sampled_indices], y[peaks][sampled_indices]

    else:
        raise ValueError("未知的平滑方法: {}".format(method))

    return x_smooth, y_smooth

def plot_training_curves(file_paths, metric="accuracy_top-1", smooth_method="moving_avg", save_path=None):
    """
    绘制训练曲线，支持不同的平滑方法
    :param file_paths: JSON 文件路径列表
    :param metric: 选择绘制的指标
    :param smooth_method: 平滑方法 (moving_avg, ema, gaussian, savgol, peaks)
    :param save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        steps, values = read_json_log(file_path, metric)
        x_smooth, y_smooth = smooth_curve(steps, values, method=smooth_method)

        plt.plot(x_smooth, y_smooth, marker='o', label=f"{file_path} ({metric}, {smooth_method})")

    plt.xlabel("Training Steps")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Training {metric.replace('_', ' ').title()} Comparison ({smooth_method})")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📁 图像已保存至: {save_path}")

    plt.show()

# 使用示例：选择不同的平滑方法
# 使用示例：
file_paths = [
    "/home/kzy/project/PartDecoder/mmdetection/work_dirs/reid_r50_fishreid_dataaug/gea1/vis_data/20250222_162347.json",
    "/home/kzy/project/PartDecoder/mmdetection/work_dirs/reid_r50_fishreid_dataaug/has-20250221_214758/vis_data/20250221_214758.json",
    "/home/kzy/project/PartDecoder/mmdetection/work_dirs/reid_r50_fishreid_dataaug/gridmask_20250221_204908/vis_data/20250221_204908.json",
    # "/home/kzy/project/PartDecoder/mmdetection/work_dirs/reid_r50_fishreid_dataaug/20250221_212049/vis_data/20250221_212049.json",
    "/home/kzy/project/PartDecoder/mmdetection/work_dirs/reid_r50_fishreid_dataaug/re/vis_data/20250222_150902.json"
    
    ]# 替换为你的 JSON 日志文件

# ✅ 1. 滑动平均 (Moving Average)
plot_training_curves(file_paths, metric="accuracy_top-1", smooth_method="peaks", save_path="accuracy_moving_avg.png")

# ✅ 2. 指数加权移动平均 (EMA)
# plot_training_curves(file_paths, metric="loss", smooth_method="ema", save_path="loss_ema.png")

# # ✅ 3. 高斯滤波 (Gaussian)
# plot_training_curves(file_paths, metric="triplet_loss", smooth_method="gaussian", save_path="triplet_loss_gaussian.png")

# # ✅ 4. Savitzky-Golay 滤波 (SG 滤波)
# plot_training_curves(file_paths, metric="ce_loss", smooth_method="savgol", save_path="ce_loss_savgol.png")

# # ✅ 5. 只选择波峰点
# plot_training_curves(file_paths, metric="accuracy_top-1", smooth_method="peaks", save_path="accuracy_peaks.png")


