import cv2

def mean_filter(image, kernel_size=3):
    """空间均值滤波"""
    return cv2.blur(image, (kernel_size, kernel_size))

def median_filter(image, kernel_size=3):
    """空间中值滤波 (kernel_size 必须为奇数)"""
    return cv2.medianBlur(image, kernel_size)