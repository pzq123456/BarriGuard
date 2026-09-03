"""红色连通域诊断：打印每个红 blob 的几何/颜色特征表，用于调红底座阈值。

有用：调 RED_* 阈值时先跑这个，看真水马红与干扰（锈铁/土坡/警示灯）的可分性。
输出写入本轮 run 目录：mask_red.png、red_components.png。
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_input, new_run_dir, save

FRAME = "1749_0944.png"


def main():
    img = load_input(FRAME)
    h, w = img.shape[:2]
    run = new_run_dir("diag_red_cc")
    print(f"run: {run}")

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    H, S, V = cv.split(hsv)
    A = lab[:, :, 1]

    # 采样结论：水马红 a*≈150-176, S 高；土坡/路面 a*≈120-130, S 低
    red_hsv = cv.inRange(H, 0, 10) | cv.inRange(H, 168, 180)
    red = red_hsv & cv.inRange(S, 60, 255) & cv.inRange(A, 145, 255)
    red = cv.morphologyEx(red, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red = cv.morphologyEx(red, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    save(red, run, "mask_red.png")

    n, labels, stats, cents = cv.connectedComponentsWithStats(red, 8)
    vis = img.copy()
    min_area = int(w * h * 3e-5)
    keep = 0
    print(f"{'id':>4}{'x':>6}{'y':>6}{'w':>5}{'h':>5}{'area':>7}{'asp':>5}{'fill':>5}"
          f"{'V':>5}{'a*':>5}")
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        m = labels[y:y + bh, x:x + bw] == i
        mv = V[y:y + bh, x:x + bw][m].mean()
        ma = A[y:y + bh, x:x + bw][m].mean()
        # 采样结论：真水马红 a*>=156 且 V>=188；锈迹/砖墙/土坡 a*<=152
        if ma < 155 or mv < 170:
            continue
        keep += 1
        print(f"{i:>4}{x:>6}{y:>6}{bw:>5}{bh:>5}{area:>7}{bw / bh:>5.2f}"
              f"{area / (bw * bh):>5.2f}{mv:>5.0f}{ma:>5.0f}")
        cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv.drawMarker(vis, (int(cents[i][0]), int(cents[i][1])), (0, 255, 255),
                      cv.MARKER_CROSS, 12, 2)
        cv.putText(vis, str(i), (x, y - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 255), 2)

    save(vis, run, "red_components.png")
    print(f"image {w}x{h}, components kept {keep} (min_area={min_area}), total {n - 1}")


if __name__ == "__main__":
    main()
