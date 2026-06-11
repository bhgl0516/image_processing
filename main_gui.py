import os
import sys

import matplotlib

# ================= 核心路径注入 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 强制注入四个实验的目录
for exp in ['experiment1', 'experiment2', 'experiment3', 'experiment4']:
    exp_path = os.path.join(BASE_DIR, exp)
    if exp_path not in sys.path:
        sys.path.insert(0, exp_path)

# 禁用 matplotlib 弹窗
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.show = lambda: None

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QPixmap, QTextCursor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QPushButton,
                             QStackedWidget, QSizePolicy, QCheckBox,
                             QGraphicsDropShadowEffect, QMessageBox,
                             QGroupBox, QFormLayout, QLineEdit, QTextEdit)

# ================= 动态导入实验入口 =================
try:
    from experiment1.run_all import main as run_exp1
except Exception as e:
    print(f"[导入警告] 实验一: {e}");
    run_exp1 = None

try:
    from experiment2.run_all_ex2 import main as run_exp2
except Exception as e:
    print(f"[导入警告] 实验二: {e}");
    run_exp2 = None

try:
    from experiment3.run_all_ex3 import main as run_exp3
except Exception as e:
    print(f"[导入警告] 实验三: {e}");
    run_exp3 = None

try:
    from experiment4.run_all_ex4 import main as run_exp4
except Exception as e:
    print(f"[导入警告] 实验四: {e}");
    run_exp4 = None

# ================= UI 样式常量 =================
FONT_FAMILY = '"Microsoft YaHei", "SimHei", sans-serif'


def apply_shadow(widget):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(15)
    shadow.setXOffset(2)
    shadow.setYOffset(4)
    shadow.setColor(QColor(0, 0, 0, 60))
    widget.setGraphicsEffect(shadow)


# ================= 动态缩放图片控件 =================
class ScalableLabel(QLabel):
    def __init__(self, text="", on_click=None):
        super().__init__(text)
        self._pixmap = None
        self.on_click = on_click
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        super().setPixmap(self.scaledPixmap())
        # 防止图片尺寸撑大右侧面板，导致左栏被挤压
        self.setMinimumSize(0, 0)

    def resizeEvent(self, event):
        if self._pixmap:
            super().setPixmap(self.scaledPixmap())
        super().resizeEvent(event)

    def scaledPixmap(self):
        return self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def mousePressEvent(self, event):
        if self._pixmap and not self._pixmap.isNull() and self.on_click:
            self.on_click(self._pixmap)
        super().mousePressEvent(event)


class StreamRedirector(QObject):
    text_written = pyqtSignal(str)

    def write(self, text): self.text_written.emit(str(text))

    def flush(self): pass


class ExperimentWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, run_func, target_folder, params_dict):
        super().__init__()
        self.run_func = run_func
        self.target_folder = target_folder
        self.params_dict = params_dict

    def run(self):
        old_cwd = os.getcwd()
        exp_dir = os.path.join(BASE_DIR, self.target_folder)
        if not os.path.exists(exp_dir): os.makedirs(exp_dir)
        os.chdir(exp_dir)
        try:
            self.run_func(**self.params_dict)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            os.chdir(old_cwd)
            self.finished.emit()


class ImageProcessingApp(QMainWindow):
    _BASE_W = 1350  # 基准窗口宽度，用于计算字体缩放系数

    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能图像处理 - 综合实验平台")
        self.setMinimumSize(1100, 750)
        self.resize(self._BASE_W, 850)

        self._scale = 1.0
        self._style_registry = {}  # id(widget) -> (widget, apply_fn(scale))

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.redirector = StreamRedirector()
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        self.init_pages()

    # ================= 字体缩放辅助 =================
    def _fs(self, px):
        """返回随窗口缩放的实际 px 值"""
        return max(8, int(px * self._scale))

    def _ss(self, base_font_size, extra_style=""):
        """注册一个带缩放功能的内联样式。
        返回最终的 stylesheet 字符串（创建时计算一次），
        同时将缩放函数注册到 _style_registry 中，方便 resize 时重建。
        用法: widget.setStyleSheet(self._ss(14, "font-weight: bold; color: #xxx;"))
        """
        fs = self._fs(base_font_size)
        return f"{extra_style}; font-size: {fs}px;" if extra_style else f"font-size: {fs}px;"

    def _register_scaled_style(self, widget, base_font_size, extra_style="",
                               style_fn=None):
        """注册一个可缩放的内联样式，resize 时自动重建。
        默认行为: 把 font-size 缩放，拼接 extra_style。
        若传入 style_fn(scale) -> str，则完全由该函数生成 stylesheet。
        """
        def _apply(sc):
            if style_fn:
                ss = style_fn(sc)
            else:
                fs = max(8, int(base_font_size * sc))
                ss = f"{extra_style}; font-size: {fs}px;" if extra_style else f"font-size: {fs}px;"
            try:
                widget.setStyleSheet(ss)
            except RuntimeError:
                pass
        self._style_registry[id(widget)] = (widget, _apply)
        _apply(self._scale)

    def _rebuild_scaled_styles(self):
        """重建所有注册过的内联样式（resize 时调用）"""
        dead = []
        for wid, (w, fn) in self._style_registry.items():
            try:
                fn(self._scale)
            except RuntimeError:
                dead.append(wid)
        for wid in dead:
            self._style_registry.pop(wid, None)

    def resizeEvent(self, event):
        """窗口缩放时更新字体缩放系数并重建所有样式"""
        super().resizeEvent(event)
        new_scale = max(0.5, min(2.0, self.width() / self._BASE_W))
        if abs(new_scale - self._scale) > 0.02:
            self._scale = new_scale
            # 重建全局 QSS
            apply_stylesheet(QApplication.instance(), self._scale)
            # 重建各注册的内联样式
            self._rebuild_scaled_styles()
            # 刷新各二级页的布局约束（左栏宽度、顶栏高度等）
            self._refresh_lab_layouts()

    def _refresh_lab_layouts(self):
        """刷新 lab 页面的布局约束（随 _scale 变化）"""
        for page in [self.lab1_page, self.lab2_page, self.lab3_page, self.lab4_page]:
            lw = getattr(page, '_left_widget', None)
            cw = getattr(page, '_content_widget', None)
            tb = getattr(page, '_top_bar', None)
            if lw and cw:
                # 左栏宽度固定为内容区宽度的 30%，确保不被右侧内容挤压
                pct_w = int(cw.width() * 0.30)
                clamped = max(self._fs(260), min(self._fs(420), pct_w))
                lw.setMinimumWidth(clamped)
                lw.setMaximumWidth(clamped + 1)
                # 同步刷新内容区间距
                ml = cw.layout()
                if ml:
                    fs20 = self._fs(20)
                    fs15 = self._fs(15)
                    ml.setContentsMargins(fs20, fs15, fs20, fs20)
                    ml.setSpacing(self._fs(20))
            if tb:
                tb.setFixedHeight(self._fs(48))

    def init_pages(self):
        self.main_page = QWidget()
        self.setup_main_page()
        self.stacked_widget.addWidget(self.main_page)

        # ================= 核心修复：传入图片列表（支持多图显示） =================
        params_exp1 = [("目标 PSNR (dB)", "28.0"), ("滤波器大小", "3, 5, 7")]
        imgs_exp1 = [os.path.join('experiment1', 'results', 'comparison', 'all_comparison.png')]

        params_exp2 = [("高斯噪声方差", "625"), ("椒盐噪声概率", "0.1")]

        def setup_exp2_extra(layout, input_dict):
            group = QGroupBox("实现模式选择")
            self._register_scaled_style(group, 14, style_fn=lambda sc:
                f"QGroupBox {{ font-weight: bold; font-size: {max(8, int(14*sc))}px; border: 1px solid #BDC3C7; border-radius: {max(2, int(5*sc))}px; margin-top: {max(4, int(10*sc))}px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
            vbox = QVBoxLayout(group)
            cb_noise = QCheckBox("噪声生成：使用存档版本 (noise_generator.py)")
            cb_filter = QCheckBox("空间滤波：使用存档版本 (spatial_filters.py)")
            for cb in [cb_noise, cb_filter]:
                self._register_scaled_style(cb, 14, "padding: 4px;")
                vbox.addWidget(cb)
            layout.addWidget(group)
            input_dict['_archive_noise'] = cb_noise
            input_dict['_archive_filters'] = cb_filter

        imgs_exp2 = [
            os.path.join('experiment2', 'results', 'comparison', 'gaussian_analysis.png'),
            os.path.join('experiment2', 'results', 'comparison', 'sp_analysis.png')
        ]

        params_exp3 = [("训练 Epochs", "10"), ("Batch Size", "128")]

        def setup_exp3_extra(layout, input_dict):
            cb_retrain = QCheckBox("重新训练模型（不勾选则使用本地已训练的模型）")
            self._register_scaled_style(cb_retrain, 14, "padding: 4px;")
            layout.addWidget(cb_retrain)
            input_dict['_retrain'] = cb_retrain

        imgs_exp3 = [os.path.join('experiment3', 'results', 'Experiment3_Full_Report.png')]

        params_exp4 = [("Otsu 平滑核", "5"), ("形态学开运算 r", "5")]
        imgs_exp4 = [
            os.path.join('experiment4', 'results', 'Traditional_Segmentation.png'),
            os.path.join('experiment4', 'results', 'DeepLearning_and_LiveTest.png')
        ]

        self.lab1_page = self.create_functional_lab_page("实验一 图像变换", 0, run_exp1, 'experiment1', params_exp1,
                                                         imgs_exp1)
        self.lab2_page = self.create_functional_lab_page("实验二 图像增强复原", 0, run_exp2, 'experiment2', params_exp2,
                                                         imgs_exp2, extra_setup_fn=setup_exp2_extra)
        self.lab3_page = self.create_functional_lab_page("实验三 CIFAR-10识别", 0, run_exp3, 'experiment3', params_exp3,
                                                         imgs_exp3, extra_setup_fn=setup_exp3_extra)
        self.lab4_page = self.create_functional_lab_page("实验四 图像分割处理", 0, run_exp4, 'experiment4', params_exp4,
                                                         imgs_exp4)

        self.stacked_widget.addWidget(self.lab1_page)
        self.stacked_widget.addWidget(self.lab2_page)
        self.stacked_widget.addWidget(self.lab3_page)
        self.stacked_widget.addWidget(self.lab4_page)

    def setup_main_page(self):
        main_layout = QVBoxLayout(self.main_page)
        main_layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel("智能图像处理实验")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        info_label = QLabel("姓名：xxxx    学号：xxxx")
        info_label.setObjectName("InfoLabel")
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)

        main_layout.addSpacing(self._fs(50))

        grid_layout = QGridLayout()
        grid_layout.setSpacing(self._fs(30))

        btn_lab1 = QPushButton("实验一：图像变换")
        btn_lab2 = QPushButton("实验二：图像增强复原")
        btn_lab3 = QPushButton("实验三：CIFAR-10物体识别")
        btn_lab4 = QPushButton("实验四：图像分割处理")

        for btn in [btn_lab1, btn_lab2, btn_lab3, btn_lab4]:
            btn.setProperty("class", "LabButton")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setMinimumHeight(self._fs(100))
            apply_shadow(btn)

        grid_layout.addWidget(btn_lab1, 0, 0)
        grid_layout.addWidget(btn_lab2, 0, 1)
        grid_layout.addWidget(btn_lab3, 1, 0)
        grid_layout.addWidget(btn_lab4, 1, 1)

        grid_container = QHBoxLayout()
        grid_container.addStretch(1)
        grid_container.addLayout(grid_layout, stretch=3)
        grid_container.addStretch(1)

        main_layout.addLayout(grid_container)
        main_layout.addSpacing(self._fs(30))

        # ================= 清理结果按钮组 =================
        cleanup_group = QGroupBox("清理结果")
        self._register_scaled_style(cleanup_group, 14, style_fn=lambda sc:
            f"QGroupBox {{ font-weight: bold; font-size: {max(8, int(14*sc))}px; border: 1px solid #BDC3C7; border-radius: {max(2, int(5*sc))}px; margin-top: {max(4, int(12*sc))}px; padding-top: {max(4, int(18*sc))}px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; }}")
        cleanup_layout = QHBoxLayout(cleanup_group)
        cleanup_layout.setSpacing(self._fs(12))

        exp_results = [
            (1, "实验一", os.path.join(BASE_DIR, 'experiment1', 'results')),
            (2, "实验二", os.path.join(BASE_DIR, 'experiment2', 'results')),
            (3, "实验三", os.path.join(BASE_DIR, 'experiment3', 'results')),
            (4, "实验四", os.path.join(BASE_DIR, 'experiment4', 'results')),
        ]

        def clean_dir(dir_path):
            if not os.path.isdir(dir_path):
                return
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass

        def make_clean_callback(desc, dirs_to_clean):
            def _clean():
                reply = QMessageBox.question(self, "确认清理",
                                             f"确定要清理 {desc} 的所有输出图片和报告吗？\n此操作不可撤销。",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply != QMessageBox.Yes:
                    return
                for d in dirs_to_clean:
                    clean_dir(d)
                QMessageBox.information(self, "清理完成", f"{desc} 的输出已清理完毕。")
            return _clean

        # 实验三的模型文件路径
        exp3_model_path = os.path.join(BASE_DIR, 'experiment3', 'data', 'weights', 'cifar10_cnn.pth')

        for idx, label, path in exp_results:
            btn = QPushButton(f"清理{label}")

            if idx == 3:
                # 实验三额外询问是否删除训练模型
                def _clean_exp3():
                    msg = "确定要清理实验三的所有输出图片和报告吗？\n此操作不可撤销。"
                    if os.path.exists(exp3_model_path):
                        msg += "\n\n是否同时删除已训练的模型文件？\n（否则下次运行将自动使用缓存的模型）"
                    reply = QMessageBox.question(self, "确认清理", msg,
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply != QMessageBox.Yes:
                        return
                    # 先清理 results 目录
                    clean_dir(path)
                    # 再询问是否删模型
                    if os.path.exists(exp3_model_path):
                        del_model = QMessageBox.question(self, "删除模型",
                                                         f"是否删除已训练的模型文件？\n{exp3_model_path}",
                                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                        if del_model == QMessageBox.Yes:
                            try:
                                os.remove(exp3_model_path)
                                # 清理空的 weights 目录
                                weights_dir = os.path.dirname(exp3_model_path)
                                if os.path.isdir(weights_dir) and not os.listdir(weights_dir):
                                    os.rmdir(weights_dir)
                                print(f"[已删除模型] {exp3_model_path}")
                            except Exception as e:
                                print(f"[删除模型失败] {e}")
                    QMessageBox.information(self, "清理完成", "实验三的输出已清理完毕。")
                btn.clicked.connect(_clean_exp3)
            else:
                btn.clicked.connect(make_clean_callback(label, [path]))

            self._register_scaled_style(btn, 13, "padding: 6px 14px; border-radius: 4px;")
            btn.setCursor(Qt.PointingHandCursor)
            cleanup_layout.addWidget(btn)

        btn_clean_all = QPushButton("一键全部清理")
        btn_clean_all.setObjectName("CleanAllButton")
        btn_clean_all.setCursor(Qt.PointingHandCursor)
        all_paths = [p for _, _, p in exp_results]
        btn_clean_all.clicked.connect(make_clean_callback("全部实验", all_paths))
        cleanup_layout.addWidget(btn_clean_all)

        cleanup_container = QHBoxLayout()
        cleanup_container.addStretch(1)
        cleanup_container.addWidget(cleanup_group, stretch=3)
        cleanup_container.addStretch(1)
        main_layout.addLayout(cleanup_container)

        main_layout.addSpacing(self._fs(20))

        exit_btn = QPushButton("退出程序")
        exit_btn.setObjectName("ExitButton")
        exit_btn.setCursor(Qt.PointingHandCursor)
        exit_btn.clicked.connect(self.close)

        exit_layout = QHBoxLayout()
        exit_layout.addStretch()
        exit_layout.addWidget(exit_btn)
        exit_layout.addStretch()
        main_layout.addLayout(exit_layout)

        btn_lab1.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        btn_lab2.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        btn_lab3.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        btn_lab4.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(4))

    def create_functional_lab_page(self, title_text, main_page_index, run_func, target_folder, params,
                                   result_img_paths, extra_setup_fn=None):
        page = QWidget()
        page_stack = QStackedWidget()

        # ==================== Page 0: 正常布局 ====================
        normal_page = QWidget()
        normal_layout = QVBoxLayout(normal_page)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        normal_layout.setSpacing(0)

        # 顶部导航栏：返回键在左上角
        top_bar = QWidget()
        top_bar.setFixedHeight(self._fs(48))
        page._top_bar = top_bar
        top_bar.setStyleSheet("background-color: #2C3E50;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(self._fs(8), 0, self._fs(12), 0)

        back_btn = QPushButton("← 返回主页")
        back_btn.setCursor(Qt.PointingHandCursor)
        self._register_scaled_style(back_btn, 15,
            "color: white; background: transparent; border: none; padding: 8px 12px;")
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(main_page_index))
        top_layout.addWidget(back_btn)
        top_layout.addStretch()

        top_title = QLabel(f"【{title_text}】")
        self._register_scaled_style(top_title, 17, "color: white; font-weight: bold;")
        top_layout.addWidget(top_title)
        top_layout.addStretch()

        normal_layout.addWidget(top_bar)

        # ==================== 内容区域 ====================
        content_widget = QWidget()
        main_h_layout = QHBoxLayout(content_widget)
        main_h_layout.setContentsMargins(self._fs(20), self._fs(15), self._fs(20), self._fs(20))
        main_h_layout.setSpacing(self._fs(20))
        normal_layout.addWidget(content_widget, stretch=1)

        # ==================== 左侧 (控制面板) ====================
        left_widget = QWidget()
        left_widget.setMinimumWidth(self._fs(320))
        left_widget.setMaximumWidth(self._fs(420))
        page._left_widget = left_widget
        page._content_widget = content_widget
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        param_group = QGroupBox("实验参数配置")
        self._register_scaled_style(param_group, 16, style_fn=lambda sc:
            f"QGroupBox {{ font-weight: bold; font-size: {max(8, int(16*sc))}px; border: 1px solid #BDC3C7; border-radius: {max(2, int(5*sc))}px; margin-top: {max(4, int(15*sc))}px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        form_layout = QFormLayout(param_group)
        form_layout.setSpacing(self._fs(15))

        input_dict = {}
        for label_text, default_val in params:
            line_edit = QLineEdit(default_val)
            self._register_scaled_style(line_edit, 15,
                "padding: 6px; border: 1px solid #ccc; border-radius: 4px;")
            form_layout.addRow(QLabel(label_text + ":"), line_edit)
            input_dict[label_text] = line_edit

        left_layout.addWidget(param_group)

        if extra_setup_fn:
            extra_setup_fn(left_layout, input_dict)

        # 运行按钮居中
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        run_btn = QPushButton("▶ 运行实验")
        run_btn.setObjectName("RunButton")
        run_btn.setCursor(Qt.PointingHandCursor)
        apply_shadow(run_btn)
        btn_layout.addWidget(run_btn)
        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        console_label = QLabel("终端输出 (Console)  —  点击可全屏查看")
        self._register_scaled_style(console_label, 14, "font-weight: bold; color: #34495E;")
        left_layout.addWidget(console_label)

        console_text = QTextEdit()
        console_text.setReadOnly(True)
        console_text.setCursor(Qt.PointingHandCursor)
        self._register_scaled_style(console_text, 13,
            "background-color: #1E1E1E; color: #2ECC71; font-family: Consolas; border-radius: 5px;")
        left_layout.addWidget(console_text, stretch=1)

        # ==================== 右侧 (多图自适应展示) ====================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(self._fs(15))

        # 动态创建 Label 列表，有几张图就创建几个 Label
        def show_img_fullscreen(pixmap):
            fs_img_content._pixmap = pixmap
            fs_img_content.setPixmap(
                pixmap.scaled(fs_img_content.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            page_stack.setCurrentIndex(2)

        img_labels = []
        for idx, _ in enumerate(result_img_paths):
            lbl = ScalableLabel("等待运行实验...\n(运行结束后，结果图将显示在此处)",
                                on_click=show_img_fullscreen)
            self._register_scaled_style(lbl, 18,
                "background-color: #EAECEE; border: 2px dashed #BDC3C7; border-radius: 10px; color: #7F8C8D;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            right_layout.addWidget(lbl)
            img_labels.append(lbl)

        main_h_layout.addWidget(left_widget, stretch=3)
        main_h_layout.addWidget(right_widget, stretch=7)

        page_stack.addWidget(normal_page)

        # ==================== Page 1: 全屏终端 ====================
        fs_page = QWidget()
        fs_layout = QVBoxLayout(fs_page)
        fs_layout.setContentsMargins(0, 0, 0, 0)
        fs_layout.setSpacing(0)

        fs_top_bar = QWidget()
        fs_top_bar.setFixedHeight(48)
        fs_top_bar.setStyleSheet("background-color: #2C3E50;")
        fs_top_layout = QHBoxLayout(fs_top_bar)
        fs_top_layout.setContentsMargins(8, 0, 12, 0)

        fs_back_btn = QPushButton("← 返回")
        fs_back_btn.setCursor(Qt.PointingHandCursor)
        self._register_scaled_style(fs_back_btn, 15,
            "color: white; background: transparent; border: none; padding: 8px 12px;")
        fs_back_btn.clicked.connect(lambda: page_stack.setCurrentIndex(0))
        fs_top_layout.addWidget(fs_back_btn)
        fs_top_layout.addStretch()

        fs_title = QLabel("终端输出 (全屏)")
        self._register_scaled_style(fs_title, 17, "color: white; font-weight: bold;")
        fs_top_layout.addWidget(fs_title)
        fs_top_layout.addStretch()

        fs_layout.addWidget(fs_top_bar)

        fs_console = QTextEdit()
        fs_console.setReadOnly(True)
        self._register_scaled_style(fs_console, 15,
            "background-color: #1E1E1E; color: #2ECC71; font-family: Consolas; border: none;")
        fs_layout.addWidget(fs_console, stretch=1)

        page_stack.addWidget(fs_page)

        # ==================== Page 2: 全屏图片 ====================
        fs_img_page = QWidget()
        fs_img_layout = QVBoxLayout(fs_img_page)
        fs_img_layout.setContentsMargins(0, 0, 0, 0)
        fs_img_layout.setSpacing(0)

        fs_img_top_bar = QWidget()
        fs_img_top_bar.setFixedHeight(48)
        fs_img_top_bar.setStyleSheet("background-color: #2C3E50;")
        fs_img_top_layout = QHBoxLayout(fs_img_top_bar)
        fs_img_top_layout.setContentsMargins(8, 0, 12, 0)

        fs_img_back = QPushButton("← 返回")
        fs_img_back.setCursor(Qt.PointingHandCursor)
        self._register_scaled_style(fs_img_back, 15,
            "color: white; background: transparent; border: none; padding: 8px 12px;")
        fs_img_back.clicked.connect(lambda: page_stack.setCurrentIndex(0))
        fs_img_top_layout.addWidget(fs_img_back)
        fs_img_top_layout.addStretch()

        fs_img_title = QLabel("图片全屏预览")
        self._register_scaled_style(fs_img_title, 17, "color: white; font-weight: bold;")
        fs_img_top_layout.addWidget(fs_img_title)
        fs_img_top_layout.addStretch()

        fs_img_layout.addWidget(fs_img_top_bar)

        fs_img_content = QLabel()
        fs_img_content.setAlignment(Qt.AlignCenter)
        fs_img_content.setStyleSheet("background-color: #1E1E1E;")
        fs_img_content._pixmap = None
        fs_img_layout.addWidget(fs_img_content, stretch=1)

        def fs_img_resize(event):
            if fs_img_content._pixmap and not fs_img_content._pixmap.isNull():
                fs_img_content.setPixmap(
                    fs_img_content._pixmap.scaled(fs_img_content.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        fs_img_content.resizeEvent = fs_img_resize

        page_stack.addWidget(fs_img_page)

        # ==================== 切换全屏终端 ====================
        def show_fs_console():
            fs_console.setPlainText(console_text.toPlainText())
            page_stack.setCurrentIndex(1)

        def on_fs_console_click(event):
            if console_text.toPlainText().strip():
                show_fs_console()
                return  # 已切换到全屏页，不再将事件传给隐藏的控件
            QTextEdit.mousePressEvent(console_text, event)

        console_text.mousePressEvent = on_fs_console_click

        # ==================== 组装 page ====================
        page_outer = QVBoxLayout(page)
        page_outer.setContentsMargins(0, 0, 0, 0)
        page_outer.addWidget(page_stack)

        # ==================== 运行逻辑 ====================
        def update_console(text):
            if self.stacked_widget.currentWidget() == page:
                console_text.moveCursor(QTextCursor.End)
                console_text.insertPlainText(text)
                console_text.moveCursor(QTextCursor.End)
                # 同步到全屏终端
                if page_stack.currentIndex() == 1:
                    fs_console.moveCursor(QTextCursor.End)
                    fs_console.insertPlainText(text)
                    fs_console.moveCursor(QTextCursor.End)

        def on_run_clicked():
            if run_func is None:
                QMessageBox.warning(self, "入口未找到",
                                    f"【{title_text}】导入失败！\n请查看终端第一行的 [导入警告] 了解具体原因。")
                return

            current_params = {}
            for k, v in input_dict.items():
                if isinstance(v, QCheckBox):
                    current_params[k] = v.isChecked()
                else:
                    current_params[k] = v.text()

            run_btn.setEnabled(False)
            back_btn.setEnabled(False)
            run_btn.setText("运行中...")
            console_text.clear()

            # 运行前重置所有图片 Label
            for lbl in img_labels:
                lbl._pixmap = None
                lbl.setText("正在后台计算，请耐心等待...\n(查看左侧控制台进度)")

            self.redirector.text_written.connect(update_console)
            print(f"========== 开始执行: {title_text} ==========\n")

            self.worker = ExperimentWorker(run_func, target_folder, current_params)
            self.worker.finished.connect(on_run_finished)
            self.worker.error.connect(on_run_error)
            self.worker.start()

        _disconnected = False  # 防止 error 和 finished 信号双发导致二次 disconnect 崩溃

        def safe_disconnect():
            nonlocal _disconnected
            if _disconnected:
                return
            _disconnected = True
            try:
                self.redirector.text_written.disconnect(update_console)
            except TypeError:
                pass  # 信号未连接时忽略

        def on_run_finished():
            print(f"\n========== {title_text} 执行完毕 ==========")
            safe_disconnect()
            run_btn.setEnabled(True)
            back_btn.setEnabled(True)
            run_btn.setText("▶ 重新运行")

            # 遍历加载所有结果图片
            for idx, img_path in enumerate(result_img_paths):
                full_img_path = os.path.join(BASE_DIR, img_path)
                if os.path.exists(full_img_path):
                    img_labels[idx].setPixmap(QPixmap(full_img_path))
                else:
                    img_labels[idx].setText(f"未找到结果图片:\n{os.path.basename(img_path)}")

        def on_run_error(err_msg):
            print(f"\n[致命错误] {err_msg}")
            safe_disconnect()
            run_btn.setEnabled(True)
            back_btn.setEnabled(True)
            run_btn.setText("▶ 重新运行")
            for lbl in img_labels:
                lbl.setText("运行发生错误，请查看控制台日志。")

        run_btn.clicked.connect(on_run_clicked)
        return page


def apply_stylesheet(app, scale=1.0):
    s = scale
    fs = lambda px: max(8, int(px * s))
    br = lambda px: max(2, int(px * s))
    pd = lambda px: max(2, int(px * s))
    qss = f"""
    QMainWindow, QWidget {{ background-color: #F8F9F9; }}
    QLabel#TitleLabel {{ font-family: {FONT_FAMILY}; font-size: {fs(48)}px; font-weight: bold; color: #2C3E50; letter-spacing: 2px; }}
    QLabel#InfoLabel {{ font-family: {FONT_FAMILY}; font-size: {fs(20)}px; color: #5D6D7E; }}
    QPushButton.LabButton {{ font-family: {FONT_FAMILY}; font-size: {fs(24)}px; font-weight: bold; background-color: #34495E; color: #ECF0F1; border: none; border-radius: {br(10)}px; }}
    QPushButton.LabButton:hover {{ background-color: #3D566E; }}
    QPushButton.LabButton:pressed {{ background-color: #1A252F; }}
    QPushButton#RunButton {{ font-family: {FONT_FAMILY}; font-size: {fs(18)}px; font-weight: bold; background-color: #27AE60; color: white; border: none; border-radius: {br(6)}px; padding: {pd(12)}px; }}
    QPushButton#RunButton:hover {{ background-color: #2ECC71; }}
    QPushButton#RunButton:pressed {{ background-color: #229954; }}
    QPushButton#RunButton:disabled {{ background-color: #7DCEA0; }}
    QPushButton#ExitButton, QPushButton#BackButton {{ font-family: {FONT_FAMILY}; font-size: {fs(16)}px; background-color: #95A5A6; color: white; border: none; border-radius: {br(6)}px; padding: {pd(12)}px {pd(30)}px; }}
    QPushButton#ExitButton:hover, QPushButton#BackButton:hover {{ background-color: #AAB7B8; }}
    QPushButton#ExitButton:pressed, QPushButton#BackButton:pressed {{ background-color: #7F8C8D; }}
    QPushButton#BackButton:disabled {{ background-color: #D5DBDB; }}
    QPushButton#CleanAllButton {{ font-family: {FONT_FAMILY}; font-size: {fs(13)}px; font-weight: bold; background-color: #E74C3C; color: white; border: none; border-radius: {br(4)}px; padding: {pd(6)}px {pd(14)}px; }}
    QPushButton#CleanAllButton:hover {{ background-color: #EC7063; }}
    QPushButton#CleanAllButton:pressed {{ background-color: #B03A2E; }}
    """
    app.setStyleSheet(qss)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    apply_stylesheet(app, window._scale)
    window.show()
    sys.exit(app.exec_())