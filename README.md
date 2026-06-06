# 智能图像处理实验平台

基于 **PyQt5** 框架开发的图像处理实验集成平台，涵盖图像变换、图像增强复原等核心算法的实现与对比分析。

---

## 项目结构

```
image_processing/
├── common/                  # 公共工具（图像加载、PSNR计算等）
├── experiment1/             # 实验一：图像变换（FFT/DCT/Hadamard/深度学习压缩）
├── experiment2/             # 实验二：图像增强复原（去噪滤波/DnCNN）
├── experiment3/             # 实验三：CIFAR-10 物体识别
├── experiment4/             # 实验四：图像分割
├── main_gui.py              # PyQt5 主界面入口
└── README.md
```

---

## 环境配置

### 依赖安装

```bash
pip install -r common/requirements.txt
```

### 依赖清单

| 库 | 用途 |
|---|---|
| PyQt5 | 图形用户界面（主菜单） |
| opencv-python | 图像 I/O 与基础操作 |
| numpy | 矩阵运算与数值计算 |
| scipy | FFT / DCT / Hadamard 变换 |
| matplotlib | 结果可视化与对比图绘制 |
| torch / torchvision | 深度学习模型加载与推理 |
| compressai | 端到端图像压缩（实验一选做） |

### Python 版本

推荐 **Python 3.8+**。

---

## 快速启动

### GUI 主界面

```bash
python main_gui.py
```

### 直接运行实验

```bash
# 实验一：图像变换与压缩
cd experiment1 && python run_all.py

# 实验二：图像增强复原
cd experiment2 && python run_all_ex2.py
```

---

## 实验说明

### 实验一：图像变换与压缩

**目标**：对比 FFT、DCT、Hadamard 三种频域变换在图像压缩中的性能差异。

| 方法 | 实现 |
|---|---|
| FFT（傅里叶变换） | `numpy.fft` |
| DCT（离散余弦变换） | `scipy.fftpack.dct` |
| Hadamard（哈达玛变换） | `scipy.linalg.hadamard` |
| 深度学习压缩（选做） | CompressAI `bmshj2018_factorized` |

**流程**：
1. 正变换 → 频谱图可视化
2. 保留 **5%** 系数（其余置零）→ 逆变换重建
3. 计算 PSNR，输出各变换重建图
4. 二分查找达成 **PSNR ≈ 28dB** 所需最少非零系数
5. 对比三种方法的计算耗时与压缩效率

**输出**：频谱图、重建图、综合对比图 `all_comparison.png`

---

### 实验二：图像增强复原

**目标**：比较传统空间域滤波与深度学习方法在不同噪声下的去噪效果。

**噪声模型**：

| 类型 | 参数 |
|---|---|
| 高斯噪声 | 均值 = 0，方差 = 625（σ = 25） |
| 椒盐噪声 | 黑白噪声各 5% |

**去噪方法**：

| 方法 | 说明 |
|---|---|
| 均值滤波 | 纯 NumPy 手工实现，核尺寸 3×3 / 5×5 / 7×7 |
| 中值滤波 | 纯 NumPy 手工实现，核尺寸 3×3 / 5×5 / 7×7 |
| DnCNN（选做） | 17 层深度残差卷积网络，预训练权重 `net.pth` |

**流程**：
1. 对原图分别添加高斯 / 椒盐噪声
2. 均值滤波 & 中值滤波（3 种核尺寸）去噪
3. DnCNN 端到端去噪
4. 计算所有方法 PSNR，红框标注传统方法最优

**输出**：去噪结果图、综合对比分析图、`psnr_report.txt` 数据报告

---

### 实验三：CIFAR-10 物体识别

计划研究基于深度学习的 CIFAR-10 十类物体分类（待实现）。

### 实验四：图像分割

计划研究图像分割算法与目标区域提取（待实现）。

---

## 公共模块

`common/utils.py` 提供以下通用函数：

| 函数 | 功能 |
|---|---|
| `load_image_gray(path, size)` | 加载灰度图，调整至 2 的整数次幂尺寸 |
| `load_image_rgb(path, size)` | 加载彩图（RGB 格式） |
| `calculate_psnr(orig, recon)` | 计算峰值信噪比 |
| `save_image(img, dir, name, cmap)` | 保存图像到指定目录 |
| `keep_top_ratio(coeffs, ratio)` | 保留绝对值最大的前 ratio 比例系数 |
| `ensure_dir(path)` | 自动创建目录 |

PSNR 公式：

\[ PSNR = 20\log_{10}\left(\frac{255}{\sqrt{MSE}}\right) \]

---

## License

本项目仅供学习与研究使用。
