"""采样点颜色表：人工指定图上坐标，打印 HSV/Lab 值，为阈值设定提供依据。

有用：换新场景/新相机后重新采样，校准 barrier_seg.py 里的阈值常量。
仅打印表格，无图像输出。
"""
import sys
from pathlib import Path

import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_input

FRAME = "1749_0944.png"

# (名称, x, y) 近似采样点，针对 1749_0944.png 标定
SAMPLES = [
    ("barrier_red_1", 320, 590),   # 近处水马红底
    ("barrier_red_2", 540, 540),   # 中景
    ("barrier_red_3", 870, 460),   # 中远景
    ("barrier_red_4", 1050, 420),  # 远景
    ("barrier_red_5", 1380, 700),  # 右下近处
    ("rust_shack", 150, 400),      # 左侧铁皮锈
    ("dirt_bank", 1550, 600),      # 右侧土坡
    ("dirt_bank2", 1580, 480),
    ("lamp_orange", 395, 435),     # 警示灯
    ("lamp_orange2", 1420, 600),
    ("house_window", 573, 255),    # 房屋窗框
    ("road", 800, 800),
]


def main():
    img = load_input(FRAME)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    print(f"{'name':<14}{'H':>6}{'S':>6}{'V':>6}{'L':>6}{'A':>6}{'B':>6}")
    for name, x, y in SAMPLES:
        ph, pl = hsv[y, x], lab[y, x]
        print(f"{name:<14}{ph[0]:>6}{ph[1]:>6}{ph[2]:>6}{pl[0]:>6}{pl[1]:>6}{pl[2]:>6}")


if __name__ == "__main__":
    main()
