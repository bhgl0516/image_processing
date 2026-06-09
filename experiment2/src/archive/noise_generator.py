import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    添加高斯噪声
    :param image: 原始灰度图像
    :param mean: 噪声均值
    :param sigma: 噪声标准差 (方差的平方根)
    """
    # 生成高斯噪声矩阵
    noise = np.random.normal(mean, sigma, image.shape)
    # 叠加噪声并截断到 [0, 255] 范围
    noisy_img = image.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_sp_noise(image, prob=0.05):
    """
    添加椒盐(脉冲)噪声
    :param image: 原始灰度图像
    :param prob: 噪声比例 (例如 0.05 表示 5% 的像素被污染)
    """
    noisy_img = np.copy(image)
    # 生成随机矩阵
    rand_mat = np.random.random(image.shape)
    # 椒噪声 (白点)
    noisy_img[rand_mat < (prob / 2)] = 255
    # 盐噪声 (黑点)
    noisy_img[(rand_mat >= (prob / 2)) & (rand_mat < prob)] = 0
    return noisy_img