import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    针对提供的 net.pth 优化的 DnCNN 结构
    所有的 Conv 层均设为 bias=False 以匹配权重文件
    """

    def __init__(self, depth=17, n_channels=64, image_channels=1):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        # 第一层: Conv + ReLU (注意：这里改为 bias=False)
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # 中间层: Conv + BN + ReLU (这里原本就是 bias=False)
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层: Conv (注意：这里也改为 bias=False)
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # 残差学习：y = x - n
        out = self.dncnn(x)
        return x - out


def run_dncnn(noisy_img, weight_path, depth=17):
    """运行 DnCNN 推理，处理 'module.' 前缀并确保 bias 匹配"""
    if not os.path.exists(weight_path):
        return None

    try:
        model = DnCNN(depth=depth, image_channels=1)

        # 加载权重
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))

        # 处理 'module.' 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        # 加载处理后的权重，不再报错
        model.load_state_dict(new_state_dict)
        model.eval()

        # 图像预处理
        img_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            restored_tensor = model(img_tensor)

        # 结果后处理
        restored_img = restored_tensor.squeeze().numpy() * 255.0
        restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)

        return restored_img

    except Exception as e:
        print(f"DnCNN 推理失败: {e}")
        return None