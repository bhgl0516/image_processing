import os
import sys

import cv2
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 确保当前实验目录在 sys.path 中（from src.xxx 需要）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from common.utils import ensure_dir

# 跨模块兼容：绝对导入 + 相对导入回退
try:
    from src.traditional import apply_median_filters, otsu_segmentation, morphological_process
    from src.deep_learning import load_mask_rcnn_model, predict_and_visualize, add_gaussian_noise
except ImportError:
    from .src.traditional import apply_median_filters, otsu_segmentation, morphological_process
    from .src.deep_learning import load_mask_rcnn_model, predict_and_visualize, add_gaussian_noise

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def main(**kwargs):
    # 从 GUI 参数中解析用户配置
    otsu_smooth_k = int(kwargs.get('Otsu 平滑核', '5'))
    morph_r_max = int(kwargs.get('形态学开运算 r', '5'))
    # 确保内核大小为奇数且有效
    if otsu_smooth_k not in (3, 5, 7):
        otsu_smooth_k = 5

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    res_dir = os.path.join(base_dir, 'results')
    ensure_dir(data_dir);
    ensure_dir(res_dir)

    # 请提前将实验图片放入 data 文件夹
    ship_path = os.path.join(data_dir, 'chuanbo.bmp')
    custom_path = os.path.join(data_dir, 'custom_photo.jpg')  # 手机拍摄的照片

    if not os.path.exists(ship_path):
        print(f"【错误】找不到图像: {ship_path}。请将指导书附带的 船舶.bmp 放入 data 文件夹。")
        return

    # ==========================================================
    # 检查要点 1：传统图像分割方法 (船舶.bmp)
    # ==========================================================
    print("\n>>> 正在执行 [传统图像分割] 分析...")
    ship_gray = cv2.imread(ship_path, cv2.IMREAD_GRAYSCALE)

    # (1) 中值滤波效果分析（使用用户指定的核大小）
    kernel_sizes = [3, 5, 7]
    smooth_results = apply_median_filters(ship_gray, kernel_sizes)
    best_smoothed = smooth_results.get(otsu_smooth_k, smooth_results[5])  # 使用用户指定的核

    # (2) Otsu 阈值分割
    otsu_thresh_val, binary_ship = otsu_segmentation(best_smoothed)

    # (3) 数学形态学处理 (圆形结构元素, 半径基于用户设定)
    morph_radii = sorted(set([1, 3, morph_r_max]))  # 始终包含 1 和 3，加上用户指定的值
    morph_results = morphological_process(binary_ship, morph_radii)

    # --- 绘制传统分割总结大图 ---
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.canvas.manager.set_window_title('实验四：传统图像分割结果')
    fig1.suptitle(f"传统分割流程 (Otsu阈值: {otsu_thresh_val})", fontsize=18, fontweight='bold')

    axes1[0, 0].imshow(ship_gray, cmap='gray');
    axes1[0, 0].set_title("1. 原始图像 (船舶.bmp)")
    axes1[0, 1].imshow(best_smoothed, cmap='gray');
    axes1[0, 1].set_title(f"2. 最优中值平滑 (k={otsu_smooth_k})")
    axes1[0, 2].imshow(binary_ship, cmap='gray');
    axes1[0, 2].set_title("3. Otsu 阈值分割 (二值化)")

    for idx, r in enumerate(morph_radii[:3]):  # 最多显示前 3 个半径结果
        axes1[1, idx].imshow(morph_results[r], cmap='gray')
        axes1[1, idx].set_title(f"{4 + idx}. 形态学优化 (圆半径 r={r})")
    for idx in range(len(morph_radii), 3):  # 剩余子图置空
        axes1[1, idx].axis('off')

    for ax in axes1.flat: ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(res_dir, 'Traditional_Segmentation.png'), dpi=300)

    # ==========================================================
    # 检查要点 2：深度学习分割方法 (Mask R-CNN)
    # ==========================================================
    print("\n>>> 正在执行 [深度学习实例分割] 分析...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mask_rcnn_model(device)

    # (1) 准备一张 COCO 风格的测试图 (如果没提供 custom_photo，用自带的替代)
    test_img_path = custom_path if os.path.exists(custom_path) else ship_path
    img_bgr = cv2.imread(test_img_path)

    # 为了防止手机图片过大导致 OOM，限制最大分辨率
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    vis_dl, valid_count = predict_and_visualize(model, img_bgr, device)
    cv2.imwrite(os.path.join(res_dir, 'DL_Segmentation_Result.jpg'), cv2.cvtColor(vis_dl, cv2.COLOR_RGB2BGR))

    # ==========================================================
    # 检查要点 3 (现场测评)：噪声强度 vs 分割准确度关系图
    # ==========================================================
    print("\n>>> 正在执行 [现场测评] 噪声强度对分割影响的分析...")
    noise_levels = [0, 15, 25, 50]
    accuracy_proxy = []  # 使用高置信度检出物体的数量作为准确度代理指标
    vis_noise_results = []

    for sigma in noise_levels:
        print(f"    - 正在测试高斯噪声 Sigma = {sigma} ...")
        noisy_img = add_gaussian_noise(img_bgr, mean=0, sigma=sigma)
        vis_res, count = predict_and_visualize(model, noisy_img, device, confidence_threshold=0.5)
        accuracy_proxy.append(count)
        vis_noise_results.append(vis_res)

    # 将检出数量归一化为百分比 (以原图检出数为 100%)
    base_count = accuracy_proxy[0] if accuracy_proxy[0] > 0 else 1
    accuracy_percentages = [(c / base_count) * 100 for c in accuracy_proxy]

    # --- 绘制深度学习与现场测评大图 ---
    fig2 = plt.figure(figsize=(18, 10))
    fig2.canvas.manager.set_window_title('实验四：深度学习分割与现场测评')
    fig2.suptitle('Mask R-CNN 实例分割 & 抗噪鲁棒性现场测评', fontsize=18, fontweight='bold')

    # 左侧：实景照片分割结果展示
    ax_dl = fig2.add_subplot(2, 2, (1, 3))
    ax_dl.imshow(vis_dl)
    ax_dl.set_title(f'手机实拍场景分割验证 (共检出 {valid_count} 个目标)', fontsize=14)
    ax_dl.axis('off')

    # 右上：展示加噪 50 时的恶劣情况
    ax_noise = fig2.add_subplot(2, 2, 2)
    ax_noise.imshow(vis_noise_results[-1])
    ax_noise.set_title(f'极端高斯噪声 (Sigma=50) 分割表现', fontsize=14)
    ax_noise.axis('off')

    # 右下：噪声强度 vs 准确度 关系曲线
    ax_curve = fig2.add_subplot(2, 2, 4)
    ax_curve.plot(noise_levels, accuracy_percentages, 'r-o', linewidth=2, markersize=8)
    ax_curve.fill_between(noise_levels, accuracy_percentages, alpha=0.2, color='red')
    ax_curve.set_title('现场测评：算法准确度与噪声强度的关系图', fontsize=14)
    ax_curve.set_xlabel('高斯噪声标准差 (Sigma)', fontsize=12)
    ax_curve.set_ylabel('相对检出准确度 (%)', fontsize=12)
    ax_curve.set_ylim(-5, 110)
    ax_curve.grid(True, linestyle='--', alpha=0.7)

    # 标出具体数值
    for i, txt in enumerate(accuracy_percentages):
        ax_curve.annotate(f"{txt:.1f}%", (noise_levels[i], accuracy_percentages[i] + 3), ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(res_dir, 'DeepLearning_and_LiveTest.png'), dpi=300)
    print(f"\n>>> 实验四运行结束！所有结果已保存至 {res_dir}")
    plt.show()


if __name__ == '__main__':
    main()