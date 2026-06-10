<p align="center">
  <h1 align="center">智能图像处理综合实验平台</h1>
  <p align="center">Intelligent Image Processing — Integrated Lab Platform</p>
  <p align="center">
    <strong>PyQt5</strong> · <strong>OpenCV</strong> · <strong>NumPy</strong> · <strong>PyTorch</strong> · <strong>Matplotlib</strong>
  </p>
  <p align="center">
    <a href="#-features">Features</a> ·
    <a href="#-architecture">Architecture</a> ·
    <a href="#-quick-start">Quick Start</a> ·
    <a href="#-experiments">Experiments</a> ·
    <a href="#-project-structure">Structure</a>
  </p>
</p>

---

## 📋 Overview

An integrated image processing laboratory platform that consolidates **four academic experiments** — image transforms, enhancement & restoration, object recognition, and segmentation — into a single PyQt5 desktop application. The platform features a multi-threaded asynchronous execution engine, real-time console redirection, and a responsive zoom-capable UI for result visualization.

Built for **Python 3.11**, leveraging both traditional computer vision (OpenCV, NumPy) and deep learning (PyTorch, TorchVision, CompressAI).

---

## ✨ Features

| Feature | Description |
|---|---|
| **Unified GUI** | Single-window lab launcher with 4 experiment pages, QStackedWidget navigation |
| **Async Execution** | QThread-based worker isolates long-running experiments from UI thread |
| **Real-time Console** | stdout/stderr redirector streams live logs into the control panel |
| **Fullscreen Viewer** | Click images or console output for full-window preview with one-click return |
| **Responsive Scaling** | Font sizes, spacing, and layout constraints scale with window resize (0.5×–2.0×) |
| **One-click Cleanup** | Clear experiment artifacts per lab or all at once from the home page |
| **Mode Switching** | Toggle between pure-NumPy handcrafted implementations and OpenCV-accelerated versions (Exp. 2) |
| **Result Visualization** | Matplotlib-generated comparison charts with PSNR annotations and optimal-method highlighting |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────┐
│  QMainWindow (ImageProcessingApp)                     │
│  ┌──────────────────────────────────────────────────┐ │
│  │  QStackedWidget (page navigator)                  │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │ │
│  │  │  Home    │ │  Exp 1   │ │  Exp 2   │ │ ...  │ │ │
│  │  │  Page    │ │  Page    │ │  Page    │ │      │ │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────┘ │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌──────────── QThread Workers ─────────────────────┐ │
│  │  ExperimentWorker → executes run_func() async    │ │
│  │  StreamRedirector → sys.stdout → pyqtSignal → UI │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Design principles:**
- **Decoupled experiments** — Each lab is a standalone package under `experiment{N}/` with its own `src/`, `data/`, and `results/`. No cross-experiment dependencies.
- **Fallback imports** — Absolute→relative import chain ensures compatibility across different execution contexts (direct run, GUI-launched, IDE debug).
- **Scalable styling** — All `font-size`, `padding`, `border-radius`, and layout margins are multiplied by a window-width-derived scale factor, updated live on `resizeEvent`.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11**
- **Windows 10/11** (primary target; Linux/macOS with PyQt5 support should work)

### Install dependencies

```bash
pip install -r common/requirements.txt
```

The environment includes:
`PyQt5` `opencv-python` `numpy` `scipy` `matplotlib` `torch` `torchvision` `compressai`

### Prepare assets

| Experiment | Required file | Location |
|---|---|---|
| Exp 1 | `lena.png` | `experiment1/data/lena.png` |
| Exp 2 | `lena.png` + `net.pth` | `experiment2/data/` |
| Exp 3 | _(auto-downloads CIFAR-10)_ | — |
| Exp 4 | `chuanbo.bmp` + optional `custom_photo.jpg` | `experiment4/data/` |

> `net.pth` is a pre-trained DnCNN weight file. Place it under `experiment2/data/weights/net.pth`.

### Launch

```bash
python main_gui.py
```

---

## 🔬 Experiments

### Experiment 1 — Image Transforms

**Goal:** Apply FFT, DCT, and Hadamard transforms; zero out 95% of coefficients; reconstruct and compare PSNR.

| Checkpoint | Description |
|---|---|
| Log-magnitude spectra | Save transformed spectra for all three methods |
| Binary-search PSNR targeting | Find minimum nonzero coefficients to reach a target PSNR (configurable via GUI, default 28.0 dB) |
| Runtime profiling | Measure and compare execution time of each transform |
| DL compression (bonus) | Compare against CompressAI neural compression |

### Experiment 2 — Image Enhancement & Restoration

**Goal:** Implement spatial mean/median filtering and DnCNN denoising; evaluate kernel-size effects.

| Checkpoint | Description |
|---|---|
| **Code compliance** | Pure NumPy sliding-window implementation (`custom_noise.py`, `custom_filters.py`) — no OpenCV wrappers for noise/filter logic |
| Gaussian noise (μ=0, σ=25) | Add noise at configurable variance |
| Salt-and-pepper noise | Configurable probability (default 0.05 each) |
| DnCNN inference | Pre-trained model loading with `DataParallel` prefix stripping |
| Optimal-kernel highlighting | Red-border annotation on the best PSNR result per method |
| **Mode toggle** | GUI checkbox to switch between pure-NumPy and OpenCV-accelerated archive versions |

### Experiment 3 — CIFAR-10 Object Recognition

**Goal:** Train a CNN on CIFAR-10; evaluate accuracy drop under heavy salt-and-pepper noise.

| Checkpoint | Description |
|---|---|
| Clean accuracy ≥ 60% | 3-layer CNN with BatchNorm achieves ~73% at epoch 10 |
| Noisy evaluation | Salt-and-pepper noise (p=0.1 each) injected into test set |
| Report chart | Auto-generated: loss curve, accuracy bar chart, prediction grid |

### Experiment 4 — Image Segmentation

**Goal:** Traditional Otsu-based ship segmentation + deployment of Mask R-CNN instance segmentation.

| Checkpoint | Description |
|---|---|
| Median filtering → Otsu | Configurable kernel size (default k=5) |
| Morphological opening | Circular structuring elements with user-specified radius |
| Mask R-CNN inference | Pre-trained on COCO, supports user-supplied custom photos |
| Noise-robustness sweep | Gaussian noise σ ∈ {0, 15, 25, 50} vs. detection accuracy curve |

---

## 📁 Project Structure

```
image_processing/
├── main_gui.py                 # Application entry — PyQt5 GUI with multi-threaded runner
├── common/
│   ├── utils.py                # PSNR, image I/O, directory helpers
│   └── requirements.txt        # Dependency manifest
│
├── experiment1/                # Image Transforms
│   ├── data/                   # lena.png
│   ├── results/                # Spectra, reconstructed images, comparison chart
│   └── src/
│       ├── traditional_transforms.py   # FFT / DCT / Hadamard + inverse
│       └── deep_compression.py         # CompressAI neural codec
│
├── experiment2/                # Enhancement & Restoration
│   ├── data/                   # lena.png, weights/net.pth
│   ├── results/                # Filtered images, analysis charts, PSNR report
│   └── src/
│       ├── custom_filters.py           # NumPy sliding-window mean/median
│       ├── custom_noise.py             # NumPy Gaussian / salt-and-pepper
│       ├── dncnn_model.py              # DnCNN architecture + weight adapter
│       └── archive/                    # OpenCV-accelerated reference implementations
│           ├── noise_generator.py
│           └── spatial_filters.py
│
├── experiment3/                # CIFAR-10 Recognition
│   ├── data/                   # (auto-downloaded CIFAR-10)
│   ├── results/                # Training curves, accuracy plots
│   └── src/
│       ├── dataset.py          # DataLoader with noise injection
│       ├── model.py            # CNN with BatchNorm
│       └── train.py            # Train / evaluate loop
│
├── experiment4/                # Segmentation
│   ├── data/                   # chuanbo.bmp, custom_photo.jpg
│   ├── results/                # Segmentation masks, noise-robustness plots
│   └── src/
│       ├── traditional.py      # Otsu threshold + morphological operations
│       └── deep_learning.py    # Mask R-CNN (TorchVision) inference
│
└── README.md
```

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| **GUI Framework** | PyQt5 (QStackedWidget, QThread, QTextEdit) |
| **Image Processing** | OpenCV 4.9, NumPy 1.26 |
| **Deep Learning** | PyTorch 2.x, TorchVision (Mask R-CNN), CompressAI |
| **Visualization** | Matplotlib, GridSpec |
| **Python** | 3.11 |

---

## 🤝 Contributing

This project was developed as an academic laboratory platform. Contributions, bug reports, and feature suggestions are welcome — feel free to open an issue or submit a pull request.

---

## 📄 License

Distributed under the BSD 3-Clause License. See `LICENSE` for more information.
