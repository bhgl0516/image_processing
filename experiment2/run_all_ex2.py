import matplotlib
import os
import sys

import numpy as np

# 无头环境兼容：在导入 pyplot 前先设 Agg 后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 确保当前实验目录在 sys.path 中（from src.xxx 需要）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from common.utils import load_image_gray, calculate_psnr, save_image, ensure_dir

# 跨模块兼容：绝对导入 + 相对导入回退
try:
    from src.archive.noise_generator import add_gaussian_noise, add_sp_noise
    from src.archive.spatial_filters import mean_filter, median_filter
    HAS_ARCHIVE_IMPL = True
except ImportError:
    HAS_ARCHIVE_IMPL = False

try:
    from src.custom_noise import add_custom_gaussian_noise, add_custom_sp_noise
    from src.custom_filters import custom_mean_filter, custom_median_filter
    from src.dncnn_model import run_dncnn
except ImportError:
    from .src.custom_noise import add_custom_gaussian_noise, add_custom_sp_noise
    from .src.custom_filters import custom_mean_filter, custom_median_filter
    from .src.dncnn_model import run_dncnn

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def get_noise_funcs(use_archive=False):
    """根据模式选择噪声生成函数（统一接口）"""
    if use_archive and HAS_ARCHIVE_IMPL:
        print("    [模式] 噪声生成 -> 存档版本 (noise_generator.py)")
        def _gn(image, mean=0, variance=625):
            return add_gaussian_noise(image, mean=mean, sigma=np.sqrt(variance))
        def _spn(image, prob_white=0.05, prob_black=0.05):
            return add_sp_noise(image, prob=prob_white + prob_black)
        return _gn, _spn
    print("    [模式] 噪声生成 -> 纯 NumPy 手写 (custom_noise.py)")
    return add_custom_gaussian_noise, add_custom_sp_noise


def get_filter_funcs(use_archive=False):
    """根据模式选择滤波器函数"""
    if use_archive and HAS_ARCHIVE_IMPL:
        print("    [模式] 空间滤波 -> 存档版本 (spatial_filters.py)")
        return mean_filter, median_filter
    print("    [模式] 空间滤波 -> 纯 NumPy 手写 (custom_filters.py)")
    return custom_mean_filter, custom_median_filter


def highlight_axes(ax):
    """为传统最优结果添加红色边框"""
    for spine in ax.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(3.5)
        spine.set_visible(True)


def run_filter_suite(noisy_img, original_img, filter_func, name, dirs):
    """测试 3, 5, 7 尺寸并记录 PSNR"""
    res = {}
    max_psnr = -1
    best_k = 3
    for k in [3, 5, 7]:
        out = filter_func(noisy_img, kernel_size=k)
        psnr = calculate_psnr(original_img, out)
        res[k] = (out, psnr)
        save_image(out, dirs, f'{name}_k{k}_{psnr:.2f}dB.png', cmap='gray')
        if psnr > max_psnr:
            max_psnr = psnr
            best_k = k
    return res, best_k


def plot_comparison_master(original, noisy, noisy_psnr, mean_res, best_k_mean,
                           med_res, best_k_med, dncnn_img, dncnn_psnr,
                           title, save_path):
    """
    3行5列 排版逻辑：
    原图 | 加噪 | 均值(3,5,7) | 中值(3,5,7) | DnCNN
    """
    fig, axes = plt.subplots(3, 5, figsize=(18, 12))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.97)

    ks = [3, 5, 7]
    for row in range(3):
        # 1. 原始图像 (居中)
        if row == 1:
            axes[row, 0].imshow(original, cmap='gray')
            axes[row, 0].set_title("1. 原始图像", fontsize=12)
        else:
            axes[row, 0].axis('off')

        # 2. 加噪图像 (居中)
        if row == 1:
            axes[row, 1].imshow(noisy, cmap='gray')
            axes[row, 1].set_title(f"2. 加噪图像\nPSNR: {noisy_psnr:.2f}dB", fontsize=12)
        else:
            axes[row, 1].axis('off')

        # 3. 均值滤波 (纵向)
        img_m, p_m = mean_res[ks[row]]
        axes[row, 2].imshow(img_m, cmap='gray')
        axes[row, 2].set_title(f"3. 均值滤波 (k={ks[row]})\n{p_m:.2f}dB", fontsize=11)
        if ks[row] == best_k_mean: highlight_axes(axes[row, 2])

        # 4. 中值滤波 (纵向)
        img_md, p_md = med_res[ks[row]]
        axes[row, 3].imshow(img_md, cmap='gray')
        axes[row, 3].set_title(f"4. 中值滤波 (k={ks[row]})\n{p_md:.2f}dB", fontsize=11)
        if ks[row] == best_k_med: highlight_axes(axes[row, 3])

        # 5. DnCNN (居中, 不标红)
        if row == 1:
            if dncnn_img is not None:
                axes[row, 4].imshow(dncnn_img, cmap='gray')
                axes[row, 4].set_title(f"5. DnCNN 复原\nPSNR: {dncnn_psnr:.2f}dB", fontsize=12)
            else:
                axes[row, 4].text(0.5, 0.5, '未加载', ha='center')
        else:
            axes[row, 4].axis('off')

    for ax in axes.flat:
        if ax.spines['top'].get_edgecolor() != 'red':
            ax.axis('off')
        else:
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
plt.show()


def main(**kwargs):
    use_archive_noise = kwargs.get('_archive_noise', False)
    use_archive_filters = kwargs.get('_archive_filters', False)
    add_gn, add_spn = get_noise_funcs(use_archive_noise)
    filt_mean, filt_median = get_filter_funcs(use_archive_filters)

    variance = float(kwargs.get('高斯噪声方差', '625'))
    prob_total = float(kwargs.get('椒盐噪声概率', '0.1'))
    prob_white = prob_total / 2.0
    prob_black = prob_total / 2.0

    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, 'data', 'lena.png')
    weight_path = os.path.join(base_dir, 'data', 'weights', 'net.pth')

    # 建立输出目录
    results_dir = os.path.join(base_dir, 'results')
    sub_dirs = ['gaussian_noise', 'sp_noise', 'comparison']
    for s in sub_dirs: ensure_dir(os.path.join(results_dir, s))

    # 准备日志文件
    report_path = os.path.join(results_dir, "psnr_report.txt")
    log_file = open(report_path, "w", encoding="utf-8")

    def log_and_print(text):
        print(text)
        log_file.write(text + "\n")

    log_and_print("=" * 60)
    log_and_print("      智能图像处理实验二：图像增强复原 实验数据报告")
    log_and_print("=" * 60 + "\n")

    img = load_image_gray(img_path, target_size=256)

    # --- 实验 1: 高斯噪声 ---
    log_and_print(f">>> 场景 A: 高斯噪声实验 (均值:0, 方差:{variance})")
    img_gn = add_gn(img, mean=0, variance=variance)
    p_gn = calculate_psnr(img, img_gn)
    log_and_print(f"    [加噪后 PSNR]: {p_gn:.2f} dB")

    res_gn_mean, bk_gn_mean = run_filter_suite(img_gn, img, filt_mean, 'mean',
                                               os.path.join(results_dir, 'gaussian_noise'))
    res_gn_med, bk_gn_med = run_filter_suite(img_gn, img, filt_median, 'median',
                                             os.path.join(results_dir, 'gaussian_noise'))

    for k in [3, 5, 7]:
        log_and_print(f"    - 均值滤波 (k={k}): {res_gn_mean[k][1]:.2f} dB {'(最优)' if k == bk_gn_mean else ''}")
        log_and_print(f"    - 中值滤波 (k={k}): {res_gn_med[k][1]:.2f} dB {'(最优)' if k == bk_gn_med else ''}")

    img_gn_dn = run_dncnn(img_gn, weight_path)
    p_gn_dn = calculate_psnr(img, img_gn_dn) if img_gn_dn is not None else 0
    log_and_print(f"    - DnCNN 深度学习: {p_gn_dn:.2f} dB\n")

    plot_comparison_master(img, img_gn, p_gn, res_gn_mean, bk_gn_mean, res_gn_med, bk_gn_med,
                           img_gn_dn, p_gn_dn, "高斯噪声复原分析 (红框代表传统方法最优)",
                           os.path.join(results_dir, 'comparison', 'gaussian_analysis.png'))

    # --- 实验 2: 椒盐噪声 ---
    log_and_print(f">>> 场景 B: 椒盐噪声实验 (总概率:{prob_total}, 各 {prob_white:.3f})")
    img_spn = add_spn(img, prob_white=prob_white, prob_black=prob_black)
    p_spn = calculate_psnr(img, img_spn)
    log_and_print(f"    [加噪后 PSNR]: {p_spn:.2f} dB")

    res_sp_mean, bk_sp_mean = run_filter_suite(img_spn, img, filt_mean, 'mean',
                                               os.path.join(results_dir, 'sp_noise'))
    res_sp_med, bk_sp_med = run_filter_suite(img_spn, img, filt_median, 'median',
                                             os.path.join(results_dir, 'sp_noise'))

    for k in [3, 5, 7]:
        log_and_print(f"    - 均值滤波 (k={k}): {res_sp_mean[k][1]:.2f} dB {'(最优)' if k == bk_sp_mean else ''}")
        log_and_print(f"    - 中值滤波 (k={k}): {res_sp_med[k][1]:.2f} dB {'(最优)' if k == bk_sp_med else ''}")

    img_sp_dn = run_dncnn(img_spn, weight_path)
    p_sp_dn = calculate_psnr(img, img_sp_dn) if img_sp_dn is not None else 0
    log_and_print(f"    - DnCNN 深度学习: {p_sp_dn:.2f} dB\n")

    plot_comparison_master(img, img_spn, p_spn, res_sp_mean, bk_sp_mean, res_sp_med, bk_sp_med,
                           img_sp_dn, p_sp_dn, "椒盐噪声复原分析 (红框代表传统方法最优)",
                           os.path.join(results_dir, 'comparison', 'sp_analysis.png'))

    log_and_print("=" * 60)
    log_and_print(f">>> 实验报告数据已同步保存至: {report_path}")
    log_file.close()


if __name__ == '__main__':
    main()