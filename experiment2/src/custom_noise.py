import numpy as np


def add_custom_gaussian_noise(image, mean=0, variance=625):
    """
    高斯噪声
    均值为 0，方差为 625 (即标准差 sigma = 25)
    """
    sigma = np.sqrt(variance)
    # 基于基础数学库生成高斯分布随机矩阵
    noise = np.random.normal(mean, sigma, image.shape)

    noisy_img = image.astype(np.float32) + noise
    # 截断到 0-255 并转回 uint8
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_custom_sp_noise(image, prob_white=0.05, prob_black=0.05):
    """
    椒盐噪声
    白色噪声点概率 0.05，黑色噪声点概率 0.05
    """
    noisy_img = np.copy(image)

    # 生成 0 到 1 之间的均匀分布随机矩阵
    rand_mat = np.random.rand(*image.shape)

    # 小于 0.05 的像素点变为黑点 (Pepper)
    noisy_img[rand_mat < prob_black] = 0

    # 大于 1 - 0.05 (即 0.95) 的像素点变为白点 (Salt)
    noisy_img[rand_mat > (1.0 - prob_white)] = 255

    return noisy_img