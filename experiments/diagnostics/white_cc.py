"""白色连通域诊断：打印每个白 blob 的几何特征表，用于调白面板阈值。

有用：调 WHITE_* 阈值时先跑这个，看白板与天空文字/窗框/路面标线的可分性。
仅打印表格，无图像输出，写入本轮 run 目录留档。
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_input, new_run_dir

FRAME = "1749_0944.png"


def main():
    img = load_input(FRAME)
    h, w = img.shape[:2]
    run = new_run_dir("diag_white_cc")
    print(f"run: {run}")

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv)
    white = cv.inRange(V, 195, 255) & cv.inRange(S, 0, 60)
    white = cv.morphologyEx(white, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv.connectedComponentsWithStats(white, 8)
    print(f"{'id':>4}{'x':>6}{'y':>6}{'w':>5}{'h':>5}{'area':>8}{'asp':>6}"
          f"{'touchTop':>9}{'touchLR':>8}")
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 300:
            continue
        tt = "Y" if y == 0 else "-"
        tl = "Y" if (x == 0 or x + bw == w) else "-"
        print(f"{i:>4}{x:>6}{y:>6}{bw:>5}{bh:>5}{area:>8}{bw / bh:>6.2f}{tt:>9}{tl:>8}")


if __name__ == "__main__":
    main()
