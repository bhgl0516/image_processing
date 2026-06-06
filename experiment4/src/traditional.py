import cv2
import numpy as np

def apply_median_filters(img, kernel_sizes=[3, 5, 7]):
    """1. 探究不同滤波窗口大小对去噪效果的影响"""
    results = {}
    for k in kernel_sizes:
        # OpenCV 的中值滤波速度极快
        filtered = cv2.medianBlur(img, k)
        results[k] = filtered
    return results

def otsu_segmentation(img):
    """2. 类间方差阈值算法 (Otsu) 获取分割图像"""
    # cv2.THRESH_OTSU 会自动计算最优阈值 ret
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, thresh

def morphological_process(binary_img, radius_list=[1, 3, 5]):
    """
    3. 数学形态学处理 (利用半径为 r 的圆形结构元素剔除细小残余点)
    使用开运算 (Opening = 先腐蚀后膨胀) 来去除前景外的孤立噪点
    """
    results = {}
    for r in radius_list:
        # 结构元素的直径(核大小) d = 2*r + 1
        d = 2 * r + 1
        # 生成圆形(椭圆)结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        # 执行开运算
        opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        results[r] = opened
    return results