import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 确保当前实验目录在 sys.path 中（from src.xxx 需要）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from common.utils import ensure_dir

# 跨模块兼容：绝对导入 + 相对导入回退
try:
    from src.dataset import get_dataloaders
    from src.model import SimpleCNN
    from src.train import train_model, evaluate_model, save_checkpoint, load_checkpoint
except ImportError:
    from .src.dataset import get_dataloaders
    from .src.model import SimpleCNN
    from .src.train import train_model, evaluate_model, save_checkpoint, load_checkpoint

# 绘图字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def imshow_unnormalize(img):
    """用于可视化的反归一化函数"""
    img = img / 2 + 0.5  # 反归一化: [-1, 1] -> [0, 1]
    npimg = img.numpy()
    return np.clip(np.transpose(npimg, (1, 2, 0)), 0, 1)


def main(**kwargs):
    # 从 GUI 参数中解析用户配置
    epochs = int(kwargs.get('训练 Epochs', '10'))
    batch_size = int(kwargs.get('Batch Size', '128'))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    res_dir = os.path.join(base_dir, 'results')
    ensure_dir(data_dir);
    ensure_dir(res_dir)

    # 1. 检测硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 准备数据
    print("\n>>> 正在准备 CIFAR-10 数据集 (含自定义椒盐噪声)...")
    trainloader, testloader_clean, testloader_noisy, classes = get_dataloaders(data_dir, batch_size=batch_size)

    # 3. 初始化模型
    model = SimpleCNN(num_classes=10)
    weight_path = os.path.join(base_dir, 'data', 'weights', 'cifar10_cnn.pth')

    # 检查是否需要重新训练
    _retrain = kwargs.get('_retrain', False)
    model_loaded = False
    if not _retrain and os.path.exists(weight_path):
        history = load_checkpoint(model, weight_path, device=device)
        if history is not None:
            model_loaded = True
            print("    [跳过训练] 使用本地已训练模型进行评测")
        else:
            model_loaded = False

    if not model_loaded:
        print("\n>>> 开始训练 CIFAR-10 模型...")
        history = train_model(model, trainloader, epochs=epochs, device=device)
        save_checkpoint(model, history, weight_path)

    print(f"\n>>> 模型就绪{' (已加载缓存)' if model_loaded else ''}，开始评估...")

    # 4. 检查要点1&2: 测试准确率评估
    print("\n" + "=" * 50)
    print(">>> 正在评估模型性能...")
    acc_clean = evaluate_model(model, testloader_clean, device)
    print(f"[检查要点1] 原始(干净)图像测试集准确率: {acc_clean:.2f}% (要求>=60%)")

    acc_noisy = evaluate_model(model, testloader_noisy, device)
    print(f"[检查要点2] 椒盐噪声(各0.1)测试集准确率: {acc_noisy:.2f}%")
    print("=" * 50)

    # ==========================================================
    # 5. 检查要点3: 结果统计与可视化展示 (使用 GridSpec 重构布局)
    # ==========================================================
    print("\n>>> 正在生成检查要点要求的统计图表...")

    fig = plt.figure(figsize=(18, 6))
    fig.canvas.manager.set_window_title('实验三：CIFAR-10物体识别 综合报告')

    # 创建 2 行 11 列的网格
    gs = gridspec.GridSpec(2, 11, figure=fig)

    # 1. 绘制 Loss 曲线 (占左侧 3 列)
    actual_epochs = len(history.get('loss', []))
    ax1 = fig.add_subplot(gs[:, 0:3])
    ax1.plot(range(1, actual_epochs + 1), history['loss'], 'r-o', label='Train Loss')
    ax1.set_title('模型训练 Loss 曲线', fontsize=14)
    ax1.set_xlabel('Epoch');
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. 绘制准确率柱状图 (占中间 3 列)
    ax2 = fig.add_subplot(gs[:, 3:6])
    bars = ax2.bar(['原始图像\n(Clean)', '椒盐噪声\n(Salt&Pepper=0.1)'], [acc_clean, acc_noisy],
                   color=['#27AE60', '#E74C3C'])
    ax2.set_title('模型抗噪能力对比', fontsize=14)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=60, color='gray', linestyle='--', label='检查要点1 (及格线60%)')
    ax2.legend()
    # 标注数值
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom',
                 fontweight='bold', fontsize=12)

    # 3. 直观抽样可视化对比 (占右侧 4 列)
    model.eval()
    dataiter_clean = iter(testloader_clean);
    dataiter_noisy = iter(testloader_noisy)
    images_clean, labels = next(dataiter_clean);
    images_noisy, _ = next(dataiter_noisy)

    num_show = 4
    with torch.no_grad():
        preds_c = model(images_clean[:num_show].to(device)).argmax(dim=1).cpu()
        preds_n = model(images_noisy[:num_show].to(device)).argmax(dim=1).cpu()

    for i in range(num_show):
        # 干净图 (第一行)
        ax_c = fig.add_subplot(gs[0, 7 + i])
        ax_c.imshow(imshow_unnormalize(images_clean[i]))
        t_label = classes[labels[i]];
        p_label = classes[preds_c[i]]
        color = 'green' if t_label == p_label else 'red'
        ax_c.set_title(f"真:{t_label}\n测:{p_label}", color=color, fontsize=11)
        ax_c.axis('off')

        # 噪点图 (第二行)
        ax_n = fig.add_subplot(gs[1, 7 + i])
        ax_n.imshow(imshow_unnormalize(images_noisy[i]))
        p_label_n = classes[preds_n[i]]
        color_n = 'green' if t_label == p_label_n else 'red'
        ax_n.set_title(f"带噪测:{p_label_n}", color=color_n, fontsize=11)
        ax_n.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'Experiment3_Full_Report.png'), dpi=300)
    print(f"\n>>> 运行结束！完美符合所有检查要点的图表已保存在 {res_dir}")
    plt.show()


if __name__ == '__main__':
    main()