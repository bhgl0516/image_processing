import numpy as np
import torch
import torchvision.transforms as transforms

# 使用 try-except 优雅处理依赖缺失
try:
    from compressai.zoo import bmshj2018_factorized
    HAS_COMPRESSAI = True
except ImportError:
    HAS_COMPRESSAI = False

def run_dl_compression(img_rgb):
    """运行基于 bmshj2018_factorized 的深度学习压缩"""
    try:
        # 初始化模型 (quality=3)
        net = bmshj2018_factorized(quality=3, pretrained=True).eval()

        # 转换为 tensor 并增加 batch 维度
        transform = transforms.ToTensor()
        x = transform(img_rgb).unsqueeze(0)

        # Padding (需为64的倍数)
        h, w = x.size(2), x.size(3)
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        # 前向推理
        with torch.no_grad():
            out_net = net(x)
            x_hat = out_net['x_hat']

        # Tensor 恢复为 numpy
        x_hat = x_hat.squeeze(0).clamp(0, 1).numpy()
        x_hat = np.transpose(x_hat, (1, 2, 0))

        # 裁剪掉 padding 并转回 0-255 区间
        x_hat = (x_hat[:h, :w, :] * 255.0).astype(np.uint8)
        return x_hat
    except Exception as e:
        print(f"深度学习模型加载或运行失败: {e}")
        return None