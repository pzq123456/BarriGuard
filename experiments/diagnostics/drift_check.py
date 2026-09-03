"""跨时颜色漂移诊断：锚定帧严格阈值 CC 像素，在目标帧读同位置分布。

用途：量化 AWB/AGC 漂移（疑点1），并实测相对特征 ExR/RGr 是否比绝对特征 a*/S 漂得少。
用法：python diagnostics/drift_check.py <锚定帧> <目标帧>
输出：控制台分布表（mean/p5/p50/p95）。
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_input

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RED_H1, RED_H2, RED_H3, RED_H4 = 0, 10, 168, 180
RED_S_MIN, RED_A_MIN = 60, 145
WHITE_V_MIN, WHITE_S_MAX = 195, 60
RED_CC_AREA, WHITE_CC_AREA = 30, 100
# 纯路面/纯土坡参考矩形（两帧均无屏障遮挡）
ROAD_RECT, SOIL_RECT = (700, 560, 1150, 740), (1480, 600, 1580, 750)

FEATURES = ("S", "V", "a*", "ExR", "RGr")


def feats(img):
    """逐像素特征字典：S/V/a*/ExR/RGr，均为 float32 全图。"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    b, g, r = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    return {"S": hsv[..., 1].astype(np.float32), "V": hsv[..., 2].astype(np.float32),
            "a*": lab[..., 1].astype(np.float32),
            "ExR": 2 * r - g - b, "RGr": (r - g) / (r + g + 1e-5)}


def strict_masks(img):
    """与 barrier_bin 同标定的严格红/白掩膜 + CC 筛选后的像素索引。"""
    f = feats(img)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hue = cv.inRange(hsv[..., 0], RED_H1, RED_H2) | cv.inRange(hsv[..., 0], RED_H3, RED_H4)
    red = hue & cv.inRange(f["S"].astype(np.uint8), RED_S_MIN, 255) \
        & cv.inRange(f["a*"].astype(np.uint8), RED_A_MIN, 255)
    white = cv.inRange(f["V"].astype(np.uint8), WHITE_V_MIN, 255) \
        & cv.inRange(f["S"].astype(np.uint8), 0, WHITE_S_MAX)
    out = []
    for m, min_area in ((red, RED_CC_AREA), (white, WHITE_CC_AREA)):
        m = cv.morphologyEx(m, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv.connectedComponentsWithStats(m, 8)
        px = np.zeros(m.shape, bool)
        for i in range(1, n):
            if stats[i, cv.CC_STAT_AREA] >= min_area:
                px |= labels == i
        out.append(px)
    return out


def rect_px(shape, rect):
    x0, y0, x1, y1 = rect
    m = np.zeros(shape[:2], bool)
    m[y0:y1, x0:x1] = True
    return m


def dist(f, px):
    v = f[px]
    return f"{v.mean():7.1f} {np.percentile(v, 5):7.1f} {np.percentile(v, 50):7.1f} {np.percentile(v, 95):7.1f}"


def main():
    if len(sys.argv) != 3:
        sys.exit("用法: python diagnostics/drift_check.py <锚定帧> <目标帧>")
    anchor, target = sys.argv[1], sys.argv[2]
    img_a, img_t = load_input(anchor), load_input(target)
    if img_t.shape != img_a.shape:
        print(f"警告: 目标帧 {img_t.shape[:2]} != 锚定帧 {img_a.shape[:2]}，重采样对齐后比较")
        img_t = cv.resize(img_t, img_a.shape[:2][::-1], interpolation=cv.INTER_AREA)
    f_a, f_t = feats(img_a), feats(img_t)
    red_px, white_px = strict_masks(img_a)
    road_px, soil_px = rect_px(img_a.shape, ROAD_RECT), rect_px(img_a.shape, SOIL_RECT)
    print(f"锚定 {anchor} -> 目标 {target}   (列: mean    p5      p50     p95)")

    for name, px in (("红底座CC", red_px), ("白面板CC", white_px),
                     ("纯路面", road_px), ("纯土坡", soil_px)):
        print(f"\n{name} ({px.sum()} px)")
        for k in FEATURES:
            print(f"  {k:4s} 锚帧 {dist(f_a[k], px)}\n       目帧 {dist(f_t[k], px)}")


if __name__ == "__main__":
    main()
