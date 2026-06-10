<p align="center">
  <h1 align="center">智能图像处理综合实验平台</h1>
  <p align="center">Intelligent Image Processing — Integrated Lab Platform</p>
  <p align="center">
    <strong>PyQt5</strong> · <strong>OpenCV</strong> · <strong>NumPy</strong> · <strong>PyTorch</strong> · <strong>Matplotlib</strong>
  </p>
  <p align="center">
    🌐 <a href="README.md">中文</a> | <a href="README.en.md">English</a>
  </p>
  <p align="center">
    <a href="#-功能特性">功能特性</a> ·
    <a href="#-架构设计">架构设计</a> ·
    <a href="#-快速开始">快速开始</a> ·
    <a href="#-实验说明">实验说明</a> ·
    <a href="#-项目结构">项目结构</a>
  </p>
</p>

---

## 📋 项目概述

一个集成的图像处理实验平台，将**四个学术实验**——图像变换、图像增强与复原、物体识别、图像分割——整合到统一的 PyQt5 桌面应用中。平台采用多线程异步执行引擎、控制台输出实时重定向，以及支持响应式缩放的 UI 布局，确保深度学习训练与推理时主界面保持流畅。

基于 **Python 3.11**，融合传统计算机视觉（OpenCV、NumPy）与深度学习（PyTorch、TorchVision、CompressAI）。

---

## ✨ 功能特性

| 特性 | 说明 |
|---|---|
| **统一图形界面** | 单窗口实验启动器，4 个实验页面通过 QStackedWidget 导航切换 |
| **异步执行** | 基于 QThread 的 Worker 将长时间运行的任务与 UI 线程隔离 |
| **实时控制台** | stdout/stderr 重定向器将日志流实时推送至控制面板 |
| **全屏查看器** | 点击图片或控制台输出即可全窗口预览，一键返回 |
| **响应式缩放** | 字体大小、间距和布局约束随窗口大小实时缩放（0.5×–2.0×） |
| **一键清理** | 首页支持逐个实验或全部清理生成的图片与报告 |
| **实现模式切换** | 实验二支持在纯 NumPy 手写实现与 OpenCV 加速版本之间切换 |
| **结果可视化** | Matplotlib 自动生成对比图表，标注 PSNR 并高亮最优方法 |

---

## 🏗 架构设计

```
┌──────────────────────────────────────────────────────┐
│  QMainWindow (ImageProcessingApp)                     │
│  ┌──────────────────────────────────────────────────┐ │
│  │  QStackedWidget (页面导航器)                      │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │ │
│  │  │  首页    │ │  实验一  │ │  实验二  │ │ ...  │ │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────┘ │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌──────────── QThread 工作线程 ────────────────────┐ │
│  │  ExperimentWorker → 异步执行 run_func()          │ │
│  │  StreamRedirector → sys.stdout → pyqtSignal → UI │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**设计原则：**
- **实验解耦** — 每个实验是独立的包 `experiment{N}/`，拥有自己的 `src/`、`data/` 和 `results/`。实验之间无依赖。
- **回退导入** — 绝对→相对导入链确保在不同执行环境（直接运行、GUI 启动、IDE 调试）下均可正常工作。
- **可缩放样式** — 所有 `font-size`、`padding`、`border-radius` 和布局间距均乘以基于窗口宽度计算的缩放系数，通过 `resizeEvent` 实时更新。

---

## 🚀 快速开始

### 环境要求

- **Python 3.11**
- **Windows 10/11**（主要目标平台；支持 PyQt5 的 Linux/macOS 也应可运行）

### 安装依赖

```bash
pip install -r common/requirements.txt
```

包含以下包：
`PyQt5` `opencv-python` `numpy` `scipy` `matplotlib` `torch` `torchvision` `compressai`

### 准备资源文件

| 实验 | 所需文件 | 存放路径 |
|---|---|---|
| 实验一 | `lena.png` | `experiment1/data/lena.png` |
| 实验二 | `lena.png` + `net.pth` | `experiment2/data/` |
| 实验三 | _（自动下载 CIFAR-10）_ | — |
| 实验四 | `chuanbo.bmp` + 可选 `custom_photo.jpg` | `experiment4/data/` |

> `net.pth` 是预训练的 DnCNN 权重文件，需放置于 `experiment2/data/weights/net.pth`。

### 启动程序

```bash
python main_gui.py
```

---

## 🔬 实验说明

### 实验一：图像变换

在首页单击 **实验一：图像变换** 进入。页面左侧可配置**目标 PSNR** 参数。

| 检查要点 | 说明 |
|---|---|
| 对数频谱图 | 输出并保存 FFT / DCT / Hadamard 三种变换的频谱图 |
| 二分查找目标 PSNR | 自动计算达到目标质量所需的最少非零系数个数 |
| 运算耗时对比 | 测量并输出三种变换的执行时间 |
| 深度学习压缩（选做） | 对比 CompressAI 神经压缩效果 |

**关键技术：** FFT / DCT / Hadamard 正逆变换、二分查找算法、CompressAI 深度学习编解码。

---

### 实验二：图像增强与复原

在首页单击 **实验二：图像增强复原** 进入。可配置**高斯噪声方差**和**椒盐噪声概率**，并通过勾选实现模式选择使用存档版本（OpenCV 加速）。

| 检查要点 | 说明 |
|---|---|
| **代码合规性** | 噪声生成和空间滤波器为纯 NumPy 手写实现，无 OpenCV 封装函数调用 |
| 高斯噪声 | 可配置方差（默认 625），添加高斯噪声并评估 PSNR |
| 椒盐噪声 | 可配置概率（默认各 0.05），添加椒盐噪声并评估 PSNR |
| DnCNN 深度学习去噪 | 加载预训练权重，支持 CPU/CPU 推理 |
| 最优核高亮 | 大图中用红框标记各传统方法中 PSNR 最优的核大小（3×3 / 5×5 / 7×7） |
| **模式切换** | GUI 复选框切换纯 NumPy 实现 ↔ OpenCV 加速存档版本 |

**关键技术：** 滑动窗口均值/中值滤波、高斯/椒盐噪声生成、DnCNN 卷积神经网络去噪。

---

### 实验三：CIFAR-10 物体识别

在首页单击 **实验三：CIFAR-10物体识别** 进入。可配置**训练轮数**和**批次大小**。

| 检查要点 | 说明 |
|---|---|
| 干净测试集准确率 ≥ 60% | 3 层 CNN + BatchNorm 约在第 10 轮达到 ~73% |
| 椒盐噪声评测 | 测试集注入概率各 0.1 的黑白椒盐噪声，评估精度下降 |
| 自动报表 | 生成 Loss 曲线、准确率柱状图、预测样本对比图 |

**关键技术：** 卷积神经网络（CNN）、Batch Normalization、CIFAR-10 数据加载与椒盐噪声注入。

---

### 实验四：图像分割

在首页单击 **实验四：图像分割处理** 进入。可配置**中值滤波核大小**和**形态学开运算半径**。

| 检查要点 | 说明 |
|---|---|
| 传统分割流程 | 中值滤波（可配置核大小）→ Otsu 自动阈值分割 → 形态学开运算 |
| Mask R-CNN 实例分割 | 基于 TorchVision 预训练模型，支持用户提供的实拍照片 |
| 抗噪鲁棒性评估 | 向图像注入 σ ∈ {0, 15, 25, 50} 高斯噪声，绘制检出准确率 vs 噪声强度曲线 |

**关键技术：** Otsu 阈值分割、圆形结构元素形态学开运算、Mask R-CNN（ResNet-50 FPN 骨干网络）。

---

## 📁 项目结构

```
image_processing/
├── main_gui.py                 # 主入口 — PyQt5 GUI + 多线程任务执行器
├── common/
│   ├── utils.py                # PSNR 计算、图像读写、目录管理等工具函数
│   └── requirements.txt        # 依赖清单
│
├── experiment1/                # 实验一：图像变换
│   ├── data/                   # lena.png
│   ├── results/                # 频谱图、重建图、综合对比图
│   └── src/
│       ├── traditional_transforms.py   # FFT / DCT / Hadamard 正逆变换
│       └── deep_compression.py         # CompressAI 神经网络编解码
│
├── experiment2/                # 实验二：图像增强与复原
│   ├── data/                   # lena.png, weights/net.pth
│   ├── results/                # 滤波结果、分析图、PSNR 报告
│   └── src/
│       ├── custom_filters.py           # NumPy 滑动窗口均值/中值滤波
│       ├── custom_noise.py             # NumPy 高斯/椒盐噪声生成
│       ├── dncnn_model.py              # DnCNN 模型定义 + 权重适配
│       └── archive/                    # OpenCV 加速参考实现
│           ├── noise_generator.py
│           └── spatial_filters.py
│
├── experiment3/                # 实验三：CIFAR-10 物体识别
│   ├── data/                   # （自动下载 CIFAR-10）
│   ├── results/                # 训练曲线、准确率图表
│   └── src/
│       ├── dataset.py          # 数据加载 + 椒盐噪声注入
│       ├── model.py            # CNN + BatchNorm
│       └── train.py            # 训练与评测循环
│
├── experiment4/                # 实验四：图像分割
│   ├── data/                   # chuanbo.bmp, custom_photo.jpg
│   ├── results/                # 分割结果、抗噪鲁棒性图
│   └── src/
│       ├── traditional.py      # Otsu 阈值 + 形态学处理
│       └── deep_learning.py    # Mask R-CNN（TorchVision）推理
│
├── README.md                   # 中文文档（默认）
├── README.en.md                # 英文文档
└── common/requirements.txt     # 依赖清单
```

---

## 🧰 技术栈

| 类别 | 技术 |
|---|---|
| **GUI 框架** | PyQt5（QStackedWidget、QThread、QTextEdit） |
| **图像处理** | OpenCV 4.9、NumPy 1.26 |
| **深度学习** | PyTorch 2.x、TorchVision（Mask R-CNN）、CompressAI |
| **可视化** | Matplotlib、GridSpec |
| **运行环境** | Python 3.11、Windows 10/11 |

---

## 🤝 贡献指南

本项目是一个学术实验平台。欢迎提交 Issue 或 Pull Request 来报告问题或提出改进建议。

---

## 📄 许可协议

本项目基于 BSD 3-Clause 许可证开源。详见 `LICENSE` 文件。
