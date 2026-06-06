import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QPushButton,
                             QStackedWidget, QSpacerItem, QSizePolicy,
                             QGraphicsDropShadowEffect, QMessageBox)

# ================= 核心路径注入 =================
# 确保能跨文件夹导入实验模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 动态导入实验入口 (即使某个实验还没写，也不会导致整个程序崩溃)
try:
    from experiment1.run_all import main as run_exp1
except ImportError:
    run_exp1 = None

try:
    from experiment2.run_all_ex2 import main as run_exp2
except ImportError:
    run_exp2 = None

try:
    from experiment3.run_all_ex3 import main as run_exp3
except ImportError:
    run_exp3 = None

# ================= UI 样式常量 =================
FONT_FAMILY = '"Microsoft YaHei", "SimHei", sans-serif'
SIZE_TITLE = "52px"
SIZE_INFO = "22px"
SIZE_LAB_BTN = "26px"
SIZE_BOTTOM_BTN = "16px"


def apply_shadow(widget):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(15)
    shadow.setXOffset(3)
    shadow.setYOffset(5)
    shadow.setColor(QColor(0, 0, 0, 80))
    widget.setGraphicsEffect(shadow)


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能图像处理 - 实验平台")
        self.resize(900, 700)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.init_pages()

    def init_pages(self):
        # 1. 创建主界面 (Index 0)
        self.main_page = QWidget()
        self.setup_main_page()
        self.stacked_widget.addWidget(self.main_page)

        # 2. 创建四个实验的交互页面 (Index 1-4)
        # 将我们导入的运行函数和对应的子目录传进去
        self.lab1_page = self.create_functional_lab_page("实验一 图像变换", 0, run_exp1, 'experiment1')
        self.lab2_page = self.create_functional_lab_page("实验二 图像增强复原", 0, run_exp2, 'experiment2')
        self.lab3_page = self.create_functional_lab_page("实验三 CIFAR-10物体识别", 0, run_exp3, 'experiment3')
        self.lab4_page = self.create_functional_lab_page("实验四 图像分割处理", 0, None, 'experiment4')

        self.stacked_widget.addWidget(self.lab1_page)
        self.stacked_widget.addWidget(self.lab2_page)
        self.stacked_widget.addWidget(self.lab3_page)
        self.stacked_widget.addWidget(self.lab4_page)

    def setup_main_page(self):
        main_layout = QVBoxLayout(self.main_page)
        main_layout.setAlignment(Qt.AlignCenter)

        # 顶部弹簧
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # --- 标题 ---
        title_label = QLabel("智能图像处理实验")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # --- 姓名与学号 ---
        info_label = QLabel("姓名：xxxx    学号：xxxx")
        info_label.setObjectName("InfoLabel")
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)

        main_layout.addSpacing(50)

        # --- 2x2 按钮网格 ---
        grid_layout = QGridLayout()
        grid_layout.setSpacing(25)

        btn_lab1 = QPushButton("实验一：图像变换")
        btn_lab2 = QPushButton("实验二：图像增强复原")
        btn_lab3 = QPushButton("实验三：CIFAR-10物体识别")
        btn_lab4 = QPushButton("实验四：图像分割处理")

        for btn in [btn_lab1, btn_lab2, btn_lab3, btn_lab4]:
            btn.setProperty("class", "LabButton")
            btn.setCursor(Qt.PointingHandCursor)
            apply_shadow(btn)

        grid_layout.addWidget(btn_lab1, 0, 0)
        grid_layout.addWidget(btn_lab2, 0, 1)
        grid_layout.addWidget(btn_lab3, 1, 0)
        grid_layout.addWidget(btn_lab4, 1, 1)

        grid_container = QHBoxLayout()
        grid_container.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        grid_container.addLayout(grid_layout)
        grid_container.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        main_layout.addLayout(grid_container)

        # 底部弹簧
        main_layout.addSpacerItem(QSpacerItem(20, 80, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # --- 退出按钮 ---
        exit_btn = QPushButton("退出程序")
        exit_btn.setObjectName("ExitButton")
        exit_btn.setCursor(Qt.PointingHandCursor)
        exit_btn.clicked.connect(self.close)
        apply_shadow(exit_btn)

        exit_layout = QHBoxLayout()
        exit_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        exit_layout.addWidget(exit_btn)
        exit_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        main_layout.addLayout(exit_layout)

        # --- 绑定页面跳转 ---
        btn_lab1.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        btn_lab2.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        btn_lab3.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        btn_lab4.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(4))

    def create_functional_lab_page(self, title_text, main_page_index, run_func, target_folder):
        """
        改造老师的 placeholder，加入实际的运行按钮和调度逻辑
        """
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel(f"【{title_text}】")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-family: {FONT_FAMILY}; font-size: {SIZE_TITLE}; color: #2C3E50; font-weight: bold;")

        instruction = QLabel(
            "点击下方按钮启动算法，请注意查看控制台输出进度\n(深度学习训练期间界面可能暂时无响应，请耐心等待)")
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setStyleSheet(f"font-family: {FONT_FAMILY}; font-size: 18px; color: #7F8C8D;")

        # --- 新增：运行实验按钮 ---
        run_btn = QPushButton("▶ 一键运行本实验")
        run_btn.setObjectName("RunButton")
        run_btn.setCursor(Qt.PointingHandCursor)
        apply_shadow(run_btn)

        def execute_experiment():
            if run_func is None:
                QMessageBox.warning(self, "错误",
                                    f"未找到 {title_text} 的入口文件！\n请检查 {target_folder} 目录下代码是否完整。")
                return

            print(f"\n========== 启动 {title_text} ==========")
            old_cwd = os.getcwd()
            exp_dir = os.path.join(BASE_DIR, target_folder)
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)

            os.chdir(exp_dir)  # 切换到对应实验目录，保证数据读取路径正确
            try:
                run_func()  # 执行实验主函数
            except Exception as e:
                QMessageBox.critical(self, "运行报错", f"实验运行中发生错误:\n{str(e)}")
            finally:
                os.chdir(old_cwd)  # 无论成功失败，切回主目录
                print(f"========== {title_text} 结束 ==========\n")

        run_btn.clicked.connect(execute_experiment)

        # --- 原有：返回主界面按钮 ---
        back_btn = QPushButton("返回主界面")
        back_btn.setObjectName("BackButton")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(main_page_index))
        apply_shadow(back_btn)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(instruction)
        layout.addSpacing(40)

        # 按钮水平布局
        btn_layout = QHBoxLayout()
        btn_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        btn_layout.addWidget(run_btn)
        btn_layout.addSpacing(30)
        btn_layout.addWidget(back_btn)
        btn_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        layout.addLayout(btn_layout)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        return page


def apply_stylesheet(app):
    """动态生成 QSS 样式表"""
    qss = f"""
    QMainWindow, QWidget {{
        background-color: #F4F6F7;
    }}

    QLabel#TitleLabel {{
        font-family: {FONT_FAMILY};
        font-size: {SIZE_TITLE};
        font-weight: bold;
        color: #2C3E50;
        letter-spacing: 2px;
    }}

    QLabel#InfoLabel {{
        font-family: {FONT_FAMILY};
        font-size: {SIZE_INFO};
        color: #5D6D7E;
        font-weight: normal;
    }}

    QPushButton.LabButton {{
        font-family: {FONT_FAMILY};
        font-size: {SIZE_LAB_BTN};
        font-weight: bold;
        background-color: #34495E;
        color: #ECF0F1;
        border: none;
        border-radius: 8px;
        padding: 25px 40px;  
        min-width: 280px;
    }}
    QPushButton.LabButton:hover {{
        background-color: #3D566E;
    }}
    QPushButton.LabButton:pressed {{
        background-color: #1A252F;
        padding-top: 27px; 
        padding-bottom: 23px;
    }}

    /* 运行实验按钮专属样式 */
    QPushButton#RunButton {{
        font-family: {FONT_FAMILY};
        font-size: 20px;
        font-weight: bold;
        background-color: #27AE60; /* 绿色代表运行 */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 15px 40px;
        min-width: 200px;
    }}
    QPushButton#RunButton:hover {{ background-color: #2ECC71; }}
    QPushButton#RunButton:pressed {{
        background-color: #229954;
        padding-top: 17px; padding-bottom: 13px;
    }}

    QPushButton#ExitButton, QPushButton#BackButton {{
        font-family: {FONT_FAMILY};
        font-size: {SIZE_BOTTOM_BTN};
        background-color: #95A5A6;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 12px 50px;
        min-width: 150px;
    }}
    QPushButton#ExitButton:hover, QPushButton#BackButton:hover {{
        background-color: #AAB7B8; 
    }}
    QPushButton#ExitButton:pressed, QPushButton#BackButton:pressed {{
        background-color: #7F8C8D;
        padding-top: 14px; 
        padding-bottom: 10px;
    }}
    """
    app.setStyleSheet(qss)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_stylesheet(app)

    window = ImageProcessingApp()
    window.show()

    sys.exit(app.exec_())