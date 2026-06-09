import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 确保当前实验目录在 sys.path 中（from src.xxx 需要）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from common.utils import load_image_gray, load_image_rgb, calculate_psnr, save_image, ensure_dir

# 跨模块兼容：绝对导入 + 相对导入回退
try:
    from src.traditional_transforms import *
    from src.deep_compression import run_dl_compression
except ImportError:
    from .src.traditional_transforms import *
    from .src.deep_compression import run_dl_compression

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def count_non_zero(matrix):
    return np.count_nonzero(matrix)


def find_target_psnr(img, coeffs, inv_transform_func, target_psnr=28.0, tol=0.1, max_iter=30):
    """二分查找法，寻找满足 target_psnr 的最少非零元素保留比例"""
    low, high = 0.0001, 1.0
    best_ratio = 1.0
    best_psnr = 0

    for _ in range(max_iter):
        mid = (low + high) / 2
        thresh_coeffs = apply_threshold(coeffs, keep_ratio=mid)
        rec_img = inv_transform_func(thresh_coeffs)
        psnr = calculate_psnr(img, rec_img)

        if abs(psnr - target_psnr) < abs(best_psnr - target_psnr):
            best_psnr = psnr
            best_ratio = mid

        if psnr < target_psnr:
            low = mid
        else:
            high = mid

        if abs(psnr - target_psnr) < tol:
            break

    final_coeffs = apply_threshold(coeffs, best_ratio)
    nnz = count_non_zero(final_coeffs)
    return best_ratio, best_psnr, nnz


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, 'data', 'lena.png')
    results_dir = os.path.join(base_dir, 'results')

    # 定义并创建分类保存的目录结构
    dirs = {
        'fft': os.path.join(results_dir, 'fft'),
        'dct': os.path.join(results_dir, 'dct'),
        'hadamard': os.path.join(results_dir, 'hadamard'),
        'deep': os.path.join(results_dir, 'deep_compress'),
        'comp': os.path.join(results_dir, 'comparison')
    }
    for d in dirs.values():
        ensure_dir(d)

    print(">>> 正在加载图像...")
    img_gray = load_image_gray(img_path, target_size=256)
    img_rgb = load_image_rgb(img_path, target_size=256)
    total_pixels = img_gray.shape[0] * img_gray.shape[1]

    # ========================================================
    # 阶段一：执行保留5%系数 并保存图片
    # ========================================================
    print("\n" + "=" * 50)
    print(">>> 95%系数置零，保留5%")
    print("=" * 50)

    keep_ratio = 0.05

    # 1. FFT
    f_spec, f_rec = fourier_transform_pipeline(img_gray, keep_ratio)
    psnr_f = calculate_psnr(img_gray, f_rec)
    save_image(f_spec, dirs['fft'], 'fft_spectrum.png', cmap='spectrum')
    save_image(f_rec, dirs['fft'], f'fft_reconstructed_{psnr_f:.2f}dB.png', cmap='gray')

    # 2. DCT
    d_spec, d_rec = dct_transform_pipeline(img_gray, keep_ratio)
    psnr_d = calculate_psnr(img_gray, d_rec)
    save_image(d_spec, dirs['dct'], 'dct_spectrum.png', cmap='spectrum')
    save_image(d_rec, dirs['dct'], f'dct_reconstructed_{psnr_d:.2f}dB.png', cmap='gray')

    # 3. Hadamard
    h_spec, h_rec = hadamard_transform_pipeline(img_gray, keep_ratio)
    psnr_h = calculate_psnr(img_gray, h_rec)
    save_image(h_spec, dirs['hadamard'], 'hadamard_spectrum.png', cmap='spectrum')
    save_image(h_rec, dirs['hadamard'], f'had_reconstructed_{psnr_h:.2f}dB.png', cmap='gray')

    # 4. Deep Learning
    dl_rec = run_dl_compression(img_rgb)
    psnr_dl = 0
    if dl_rec is not None:
        psnr_dl = calculate_psnr(img_rgb, dl_rec)
        dl_rec_bgr = cv2.cvtColor(dl_rec, cv2.COLOR_RGB2BGR)  # 保存用 BGR
        save_image(dl_rec_bgr, dirs['deep'], f'deep_reconstructed_{psnr_dl:.2f}dB.png', cmap='gray')

    print(f"[FFT]  保留 5% 逆变换 PSNR: {psnr_f:.2f} dB")
    print(f"[DCT]  保留 5% 逆变换 PSNR: {psnr_d:.2f} dB")
    print(f"[HAD]  保留 5% 逆变换 PSNR: {psnr_h:.2f} dB")
    if dl_rec is not None:
        print(f"[Deep] 深度学习直接重建 PSNR: {psnr_dl:.2f} dB")


    # 阶段二：找28dB对应系数，及计算耗时

    print("\n" + "=" * 50)
    print(">>>  寻找 PSNR≈28dB 最少非零元素 & 耗时对比")
    print("=" * 50)

    # FFT 测速与查参
    t0 = time.perf_counter()
    fft_coeffs = get_fft_coeffs(img_gray)
    _ = inv_fft(fft_coeffs)
    time_fft = time.perf_counter() - t0
    r_f, p_f, n_f = find_target_psnr(img_gray, fft_coeffs, inv_fft, target_psnr=28.0)

    # DCT 测速与查参
    t0 = time.perf_counter()
    dct_coeffs = get_dct_coeffs(img_gray)
    _ = inv_dct(dct_coeffs)
    time_dct = time.perf_counter() - t0
    r_d, p_d, n_d = find_target_psnr(img_gray, dct_coeffs, inv_dct, target_psnr=28.0)

    # HAD 测速与查参
    t0 = time.perf_counter()
    had_coeffs = get_hadamard_coeffs(img_gray)
    _ = inv_hadamard(had_coeffs)
    time_had = time.perf_counter() - t0
    r_h, p_h, n_h = find_target_psnr(img_gray, had_coeffs, inv_hadamard, target_psnr=28.0)

    print(f"FFT 耗时: {time_fft:.4f}s | DCT 耗时: {time_dct:.4f}s | HAD 耗时: {time_had:.4f}s")
    print("-" * 50)
    print(f"达成 PSNR ≈ 28dB (总像素: {total_pixels}) 所需非零元素：")
    print(f"       FFT 需要保留 {n_f} 个 (约 {r_f * 100:.2f}%)")
    print(f"       DCT 需要保留 {n_d} 个 (约 {r_d * 100:.2f}%)")
    print(f"       HAD 需要保留 {n_h} 个 (约 {r_h * 100:.2f}%)")
    print("=" * 50)


    # 绘制、保存纯净版综合对比图

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    fig.canvas.manager.set_window_title('实验一：图像变换结果 (保留5%系数)')

    # FFT
    axes[0, 0].imshow(img_gray, cmap='gray');
    axes[0, 0].set_title('原始图像')
    axes[0, 1].imshow(f_spec, cmap='gray');
    axes[0, 1].set_title('FFT 频谱图')
    axes[0, 2].imshow(f_rec, cmap='gray');
    axes[0, 2].set_title(f'FFT 重建图\nPSNR: {psnr_f:.2f}dB')

    # DCT
    axes[1, 0].imshow(img_gray, cmap='gray');
    axes[1, 0].set_title('原始图像')
    axes[1, 1].imshow(d_spec, cmap='gray');
    axes[1, 1].set_title('DCT 频谱图')
    axes[1, 2].imshow(d_rec, cmap='gray');
    axes[1, 2].set_title(f'DCT 重建图\nPSNR: {psnr_d:.2f}dB')

    # HAD
    axes[2, 0].imshow(img_gray, cmap='gray');
    axes[2, 0].set_title('原始图像')
    axes[2, 1].imshow(h_spec, cmap='gray');
    axes[2, 1].set_title('HAD 频谱图')
    axes[2, 2].imshow(h_rec, cmap='gray');
    axes[2, 2].set_title(f'HAD 重建图\nPSNR: {psnr_h:.2f}dB')

    # 深度学习选做题
    if dl_rec is not None:
        axes[0, 3].imshow(dl_rec);
        axes[0, 3].set_title(f'[选做] 深度学习重建\nPSNR: {psnr_dl:.2f}dB')
    else:
        axes[0, 3].text(0.5, 0.5, 'DL 未运行', ha='center', fontsize=12)

    axes[1, 3].axis('off')
    axes[2, 3].axis('off')

    for ax in axes.flat:
        if not ax.get_title() == '':
            ax.axis('off')

    plt.tight_layout()
    # 保存整合图到 comparison 文件夹
    plt.savefig(os.path.join(dirs['comp'], 'all_comparison.png'), dpi=300)
    print(f">>> 实验结果已分类保存至 {results_dir} 目录下")
    plt.show()


if __name__ == '__main__':
    main()