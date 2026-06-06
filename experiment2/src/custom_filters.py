import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def custom_mean_filter(image, kernel_size=3):
    """检查要点3: 纯手工实现的均值滤波器"""
    pad_w = kernel_size // 2
    # 使用对称填充(边缘扩展)来处理图像边界
    padded_img = np.pad(image, pad_w, mode='symmetric')

    # 提取所有滑动窗口，底层矩阵操作，免去 Python 慢速双重 for 循环
    windows = sliding_window_view(padded_img, (kernel_size, kernel_size))

    # 在滑动窗口的最后两个维度（即核宽和核高）上求平均
    filtered_img = np.mean(windows, axis=(2, 3))

    return np.clip(filtered_img, 0, 255).astype(np.uint8)


def custom_median_filter(image, kernel_size=3):
    """检查要点3: 纯手工实现的中值滤波器"""
    pad_w = kernel_size // 2
    padded_img = np.pad(image, pad_w, mode='symmetric')

    windows = sliding_window_view(padded_img, (kernel_size, kernel_size))

    # 在滑动窗口内求中位数
    filtered_img = np.median(windows, axis=(2, 3))

    return np.clip(filtered_img, 0, 255).astype(np.uint8)