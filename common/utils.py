import os

import cv2
import numpy as np


def ensure_dir(dir_path):
    """确保目录存在，不存在则自动创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_image_gray(image_path, target_size=256):
    """加载灰度图像，并强制调整为 2的整数次幂 (哈达玛变换要求)"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到图像文件: {image_path}")
    img = cv2.resize(img, (target_size, target_size))
    return img


def load_image_rgb(image_path, target_size=256):
    """加载RGB图像 (用于深度学习预训练模型)"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"找不到图像文件: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    return img


def calculate_psnr(img_original, img_reconstructed):
    """计算峰值信噪比(PSNR)"""
    mse = np.mean((img_original.astype(np.float64) - img_reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def save_image(img, save_dir, filename, cmap=None):
    """保存图像到指定目录"""
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, filename)

    if cmap == 'gray':
        cv2.imwrite(save_path, img)
    else:
        # 针对归一化的频谱图进行 0-255 拉伸
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(save_path, img_normalized)


def count_non_zero(matrix):
    """统计矩阵中非零元素的个数"""
    return np.count_nonzero(matrix)

def keep_top_ratio(coeffs, ratio):
    """保留矩阵中绝对值最大的前 ratio 比例的系数，其余置 0"""
    if ratio >= 1.0: return coeffs
    if ratio <= 0.0: return np.zeros_like(coeffs)
    # 计算阈值
    threshold = np.percentile(np.abs(coeffs), 100 * (1 - ratio))
    coeffs_thresh = coeffs.copy()
    coeffs_thresh[np.abs(coeffs_thresh) < threshold] = 0
    return coeffs_thresh