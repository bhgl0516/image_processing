import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import hadamard


# ========================================================
# 第一部分：通用工具（对应核心要求 3）
# ========================================================
def apply_threshold(coeffs, keep_ratio=0.05):
    """
    核心要求(3)：设定门限，将 (1 - keep_ratio) 比例的系数（较小的值）置为 0。
    默认 keep_ratio = 0.05，即 95% 置为 0。
    """
    if keep_ratio >= 1.0: return coeffs
    if keep_ratio <= 0.0: return np.zeros_like(coeffs)

    threshold = np.percentile(np.abs(coeffs), 100 * (1 - keep_ratio))
    coeffs_thresh = coeffs.copy()
    coeffs_thresh[np.abs(coeffs_thresh) < threshold] = 0
    return coeffs_thresh


# ========================================================
# 第二部分：完整流水线 (严格对应核心要求的 1,2,3,4,5 步骤)
# ========================================================
def fourier_transform_pipeline(img, keep_ratio=0.05):
    """傅立叶变换完整流程：变换 -> 频谱 -> 置零 -> 逆变换"""
    # 1. 正变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # 2. 获取频谱 (对数显示)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    # 3. 截断保留
    f_thresh = apply_threshold(f, keep_ratio)
    # 4. 逆变换
    img_back = np.abs(np.fft.ifft2(f_thresh))
    return magnitude_spectrum, img_back


def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


def dct_transform_pipeline(img, keep_ratio=0.05):
    """DCT变换完整流程：变换 -> 频谱 -> 置零 -> 逆变换"""
    dct_coeffs = dct2(img)
    dct_spectrum = np.log(np.abs(dct_coeffs) + 1)

    dct_thresh = apply_threshold(dct_coeffs, keep_ratio)
    img_back = idct2(dct_thresh)
    return dct_spectrum, img_back


def hadamard_transform_pipeline(img, keep_ratio=0.05):
    """哈达玛变换完整流程：变换 -> 频谱 -> 置零 -> 逆变换"""
    N = img.shape[0]
    H = hadamard(N)

    hadamard_coeffs = np.dot(np.dot(H, img), H) / N
    hadamard_spectrum = np.log(np.abs(hadamard_coeffs) + 1)

    hadamard_thresh = apply_threshold(hadamard_coeffs, keep_ratio)
    img_back = np.dot(np.dot(H, hadamard_thresh), H) / N
    return hadamard_spectrum, img_back


# ========================================================
# 第三部分：原子化算子 (专为"检查要点"中的二分查找和测速准备)
# ========================================================
def get_fft_coeffs(img): return np.fft.fft2(img)


def inv_fft(coeffs): return np.abs(np.fft.ifft2(coeffs))


def get_dct_coeffs(img): return dct2(img)


def inv_dct(coeffs): return idct2(coeffs)


def get_hadamard_coeffs(img):
    H = hadamard(img.shape[0])
    return np.dot(np.dot(H, img), H) / img.shape[0]


def inv_hadamard(coeffs):
    H = hadamard(coeffs.shape[0])
    return np.dot(np.dot(H, coeffs), H) / coeffs.shape[0]