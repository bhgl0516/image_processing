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
                             QGraphicsDropShadowEffect, QMessageBox, QGroupBox,
                             QFormLayout, QLineEdit, QTextEdit)

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
    def __init__(self, text=""):
        super().__init__(text)
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        super().setPixmap(self.scaledPixmap())

    def resizeEvent(self, event):
        if self._pixmap:
            super().setPixmap(self.scaledPixmap())
        super().resizeEvent(event)

    def scaledPixmap(self):
        return self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)


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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能图像处理 - 综合实验平台")
        self.setMinimumSize(1100, 750)
        self.resize(1350, 850)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.redirector = StreamRedirector()
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        self.init_pages()

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
            group.setStyleSheet(
                "QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #BDC3C7; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
            vbox = QVBoxLayout(group)
            cb_noise = QCheckBox("噪声生成：使用存档版本 (noise_generator.py)")
            cb_filter = QCheckBox("空间滤波：使用存档版本 (spatial_filters.py)")
            for cb in [cb_noise, cb_filter]:
                cb.setStyleSheet("font-size: 14px; padding: 4px;")
                vbox.addWidget(cb)
            layout.addWidget(group)
            input_dict['_archive_noise'] = cb_noise
            input_dict['_archive_filters'] = cb_filter

        imgs_exp2 = [
            os.path.join('experiment2', 'results', 'comparison', 'gaussian_analysis.png'),
            os.path.join('experiment2', 'results', 'comparison', 'sp_analysis.png')
        ]

        params_exp3 = [("训练 Epochs", "10"), ("Batch Size", "128")]
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
                                                         imgs_exp3)
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

        main_layout.addSpacing(50)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(30)

        btn_lab1 = QPushButton("实验一：图像变换")
        btn_lab2 = QPushButton("实验二：图像增强复原")
        btn_lab3 = QPushButton("实验三：CIFAR-10物体识别")
        btn_lab4 = QPushButton("实验四：图像分割处理")

        for btn in [btn_lab1, btn_lab2, btn_lab3, btn_lab4]:
            btn.setProperty("class", "LabButton")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setMinimumHeight(100)
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
        main_layout.addSpacing(50)

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
        main_h_layout = QHBoxLayout(page)
        main_h_layout.setContentsMargins(20, 20, 20, 20)
        main_h_layout.setSpacing(20)

        # ==================== 左侧 (控制面板) ====================
        left_widget = QWidget()
        left_widget.setMinimumWidth(350)
        left_widget.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel(f"【{title_text}】")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-family: {FONT_FAMILY}; font-size: 26px; color: #2C3E50; font-weight: bold;")
        left_layout.addWidget(title)

        param_group = QGroupBox("实验参数配置")
        param_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 16px; border: 1px solid #BDC3C7; border-radius: 5px; margin-top: 15px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        form_layout = QFormLayout(param_group)
        form_layout.setSpacing(15)

        input_dict = {}
        for label_text, default_val in params:
            line_edit = QLineEdit(default_val)
            line_edit.setStyleSheet("padding: 6px; font-size: 15px; border: 1px solid #ccc; border-radius: 4px;")
            form_layout.addRow(QLabel(label_text + ":"), line_edit)
            input_dict[label_text] = line_edit

        left_layout.addWidget(param_group)

        if extra_setup_fn:
            extra_setup_fn(left_layout, input_dict)

        btn_layout = QHBoxLayout()
        run_btn = QPushButton("▶ 运行实验")
        run_btn.setObjectName("RunButton")
        run_btn.setCursor(Qt.PointingHandCursor)
        apply_shadow(run_btn)

        back_btn = QPushButton("返回主页")
        back_btn.setObjectName("BackButton")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(main_page_index))
        apply_shadow(back_btn)

        btn_layout.addWidget(run_btn)
        btn_layout.addWidget(back_btn)
        left_layout.addLayout(btn_layout)

        console_label = QLabel("终端输出 (Console):")
        console_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495E;")
        left_layout.addWidget(console_label)

        console_text = QTextEdit()
        console_text.setReadOnly(True)
        console_text.setStyleSheet(
            "background-color: #1E1E1E; color: #2ECC71; font-family: Consolas; font-size: 13px; border-radius: 5px;")
        left_layout.addWidget(console_text)

        # ==================== 右侧 (多图自适应展示) ====================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)

        # 动态创建 Label 列表，有几张图就创建几个 Label
        img_labels = []
        for _ in result_img_paths:
            lbl = ScalableLabel("等待运行实验...\n(运行结束后，结果图将显示在此处)")
            lbl.setStyleSheet(
                "background-color: #EAECEE; border: 2px dashed #BDC3C7; border-radius: 10px; font-size: 18px; color: #7F8C8D;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            right_layout.addWidget(lbl)
            img_labels.append(lbl)

        main_h_layout.addWidget(left_widget, stretch=3)
        main_h_layout.addWidget(right_widget, stretch=7)

        # ==================== 运行逻辑 ====================
        def update_console(text):
            if self.stacked_widget.currentWidget() == page:
                console_text.moveCursor(QTextCursor.End)
                console_text.insertPlainText(text)
                console_text.moveCursor(QTextCursor.End)

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


def apply_stylesheet(app):
    qss = f"""
    QMainWindow, QWidget {{ background-color: #F8F9F9; }}
    QLabel#TitleLabel {{ font-family: {FONT_FAMILY}; font-size: 48px; font-weight: bold; color: #2C3E50; letter-spacing: 2px; }}
    QLabel#InfoLabel {{ font-family: {FONT_FAMILY}; font-size: 20px; color: #5D6D7E; }}
    QPushButton.LabButton {{ font-family: {FONT_FAMILY}; font-size: 24px; font-weight: bold; background-color: #34495E; color: #ECF0F1; border: none; border-radius: 10px; }}
    QPushButton.LabButton:hover {{ background-color: #3D566E; }}
    QPushButton.LabButton:pressed {{ background-color: #1A252F; }}
    QPushButton#RunButton {{ font-family: {FONT_FAMILY}; font-size: 18px; font-weight: bold; background-color: #27AE60; color: white; border: none; border-radius: 6px; padding: 12px; }}
    QPushButton#RunButton:hover {{ background-color: #2ECC71; }}
    QPushButton#RunButton:pressed {{ background-color: #229954; }}
    QPushButton#RunButton:disabled {{ background-color: #7DCEA0; }}
    QPushButton#ExitButton, QPushButton#BackButton {{ font-family: {FONT_FAMILY}; font-size: 16px; background-color: #95A5A6; color: white; border: none; border-radius: 6px; padding: 12px 30px; }}
    QPushButton#ExitButton:hover, QPushButton#BackButton:hover {{ background-color: #AAB7B8; }}
    QPushButton#ExitButton:pressed, QPushButton#BackButton:pressed {{ background-color: #7F8C8D; }}
    QPushButton#BackButton:disabled {{ background-color: #D5DBDB; }}
    """
    app.setStyleSheet(qss)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_stylesheet(app)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())