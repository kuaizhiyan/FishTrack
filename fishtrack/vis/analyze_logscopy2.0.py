# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import argrelextrema, savgol_filter
from scipy.ndimage import gaussian_filter1d

# 读取JSON训练日志并返回step和metric数据
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

# 曲线平滑函数
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
        sampled_indices = np.linspace(0, len(peaks) - 1, min(len(peaks), 20), dtype=int)
        x_smooth, y_smooth = x[peaks][sampled_indices], y[peaks][sampled_indices]

    else:
        raise ValueError("未知的平滑方法: {}".format(method))

    return x_smooth, y_smooth


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        if not all_times:
            raise KeyError(
                'Please reduce the log interval in the config so that'
                'interval is less than iterations of one epoch.')
        epoch_ave_time = np.array(list(map(lambda x: np.mean(x), all_times)))
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f} s/iter')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f} s/iter')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(epoch_ave_time):.4f} s/iter\n')


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)

    # 设置图像大小，拉长 x 轴
    plt.figure(figsize=(10, 5))  # 让整个图形变宽
    
    # 颜色映射
    colors = {
    "GEA": "red",
    "CutOut": "blue",
    "GridMask": "green",
    "Hide-and-Seek": "orange",
    "Random Erasing": "purple"
    }

    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    # TODO: support dynamic eval interval(e.g. RTMDet) when plotting mAP.
    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[int(args.eval_interval) - 1]]:
                if 'mAP' in metric:
                    raise KeyError(
                        f'{args.json_logs[i]} does not contain metric '
                        f'{metric}. Please check if "--no-validate" is '
                        'specified when you trained the model. Or check '
                        f'if the eval_interval {args.eval_interval} in args '
                        'is equal to the eval_interval during training.')
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}. '
                    'Please reduce the log interval in the config so that '
                    'interval is less than iterations of one epoch.')

            # 获取原始数据
            xs = []
            ys = []
            for epoch in epochs:
                iters = log_dict[epoch]['step']
                xs.append(np.array(iters))
                ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)

            # 如果需要平滑，则应用平滑方法
            if args.smooth:
                xs, ys = smooth_curve(xs, ys, method=args.smooth, window_size=args.window_size)

            plt.xlabel('iter')
            plt.plot(xs, ys, label=legend[i * num_metrics + j], 
            color=colors.get(legend[i * num_metrics + j], "black"), linewidth=0.5)
            plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['bbox_mAP'],
        help='the metric that you want to plot')
    parser_plt.add_argument(
        '--start-epoch',
        type=str,
        default='1',
        help='the epoch that you want to start')
    parser_plt.add_argument(
        '--eval-interval',
        type=str,
        default='1',
        help='the eval interval when training')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)
    
    # 新增的平滑参数
    parser_plt.add_argument(
        '--smooth',
        type=str,
        choices=["moving_avg", "ema", "gaussian", "savgol", "peaks"],
        default=None,
        help="Method for smoothing the curve")
    parser_plt.add_argument(
        '--window-size',
        type=int,
        default=10,
        help="Window size for smoothing (used in moving average, EMA, etc.)")
    parser_plt.add_argument(
        '--metric',
        type=str,
        default='accuracy_top-1',
        help="Metric to be smoothed")


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='whether include outliers (default: False)')


def parse_args():
    parser = argparse.ArgumentParser(description='Plot curves or calculate train time')
    subparsers = parser.add_subparsers()
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    log_dicts = []

    for json_log in args.json_logs:
        print(f'loading {json_log}...')
        steps, values = read_json_log(json_log)
        log_dict = defaultdict(lambda: defaultdict(list))
        for step, value in zip(steps, values):
            epoch = step // 1000  # Assuming each epoch is 1000 steps
            log_dict[epoch]["step"].append(step)
            log_dict[epoch]["accuracy_top-1"].append(value)  # Use the default metric
        log_dicts.append(log_dict)

    if hasattr(args, 'smooth') and args.smooth:
        plot_curve(log_dicts, args)
    else:
        cal_train_time(log_dicts, args)


if __name__ == '__main__':
    main()
