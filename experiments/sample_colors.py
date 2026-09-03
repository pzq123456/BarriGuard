import cv2 as cv
import numpy as np

img = cv.imread("tmp/image.png")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

# (名称, x, y) 近似采样点
samples = [
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

print(f"{'name':<14}{'H':>6}{'S':>6}{'V':>6}{'L':>6}{'A':>6}{'B':>6}")
for name, x, y in samples:
    px_hsv = hsv[y, x]
    px_lab = lab[y, x]
    print(f"{name:<14}{px_hsv[0]:>6}{px_hsv[1]:>6}{px_hsv[2]:>6}{px_lab[0]:>6}{px_lab[1]:>6}{px_lab[2]:>6}")
