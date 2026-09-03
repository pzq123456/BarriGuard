"""水马语义二值化核心：移植自 experiments/barrier_bin.py（验收通过版本）。

红∪白粗提取 × 近严远宽双阈值 × ROI 几何约束 × 定向闭运算连成屏障带 × 连通域清杂。
标定常量定义在 1622x905，按帧分辨率等比缩放，帧本身不做 resize。
"""
import cv2 as cv
import numpy as np

from . import config as cfg


def kernels(w, h):
    """形态学核与最小面积：从标定值等比缩放到当前分辨率。"""
    sx, sy = w / cfg.CALIB_W, h / cfg.CALIB_H
    k = lambda kw, kh: (max(1, round(kw * sx)), max(1, round(kh * sy)))
    area = lambda a: max(1, round(a * sx * sy))
    return (k(*cfg.OPEN_K), k(*cfg.CLOSE_K), k(*cfg.KNIT_H), k(*cfg.KNIT_V),
            area(cfg.MIN_CC_AREA), area(cfg.FAR_MIN_CC_AREA))


def rect_mask(shape, rects, val=255):
    m = np.zeros(shape[:2], np.uint8)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = val
    return m


def color_masks(hsv, lab, s_red, a_red, v_red, v_white, s_white):
    """一组(红,白)掩膜；远端版多一个 V 下限防暗色噪声。"""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = cv.inRange(h, cfg.RED_H1, cfg.RED_H2) | cv.inRange(h, cfg.RED_H3, cfg.RED_H4)
    red = hue & cv.inRange(s, s_red, 255) & cv.inRange(lab[..., 1], a_red, 255)
    if v_red:
        red &= cv.inRange(v, v_red, 255)
    white = cv.inRange(v, v_white, 255) & cv.inRange(s, 0, s_white)
    return red, white


def build_fg(hsv, lab, far_band, k_open, k_close, k_knit_h, k_knit_v):
    """近严远宽粗提取 + 定向闭运算，把红底座和白板抹平成连续屏障带。"""
    red, white = color_masks(hsv, lab, cfg.RED_S_MIN, cfg.RED_A_MIN, 0,
                             cfg.WHITE_V_MIN, cfg.WHITE_S_MAX)
    red_f, white_f = color_masks(hsv, lab, cfg.FAR_RED_S_MIN, cfg.FAR_RED_A_MIN,
                                 cfg.FAR_RED_V_MIN, cfg.WHITE_V_MIN, cfg.WHITE_S_MAX)
    for m in (red, white, red_f, white_f):
        m[:] = cv.morphologyEx(m, cv.MORPH_OPEN, np.ones(k_open, np.uint8))
    fg = (red | white) | ((red_f | white_f) & far_band)
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(k_close, np.uint8))
    # 线段核沿路向收编断缝：横向成排 + 纵向红白堆叠
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(k_knit_h, np.uint8))
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(k_knit_v, np.uint8))
    return fg


def cc_mean(hsv, lab, labels, i, x, y, w, h):
    m = labels[y:y + h, x:x + w] == i
    return (lab[..., 1][y:y + h, x:x + w][m].mean(),
            hsv[..., 1][y:y + h, x:x + w][m].mean(),
            hsv[..., 2][y:y + h, x:x + w][m].mean())


def clean(fg, hsv, lab, far_band, min_cc, min_cc_far):
    """连通域清杂：小斑剔除；均值色既不偏红也不偏白的（灰路面/树影）剔除。"""
    n, labels, stats, _ = cv.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        x, y, w, h, area = stats[i, :5]
        if area < (min_cc_far if far_band[y + h // 2, x + w // 2] else min_cc):
            continue
        a, s, v = cc_mean(hsv, lab, labels, i, x, y, w, h)
        if a >= cfg.CC_A_REDISH or (v >= cfg.CC_V_WHITEISH and s <= cfg.CC_S_WHITEISH):
            out[labels == i] = 255
    return out


def process(frame: np.ndarray, cam: cfg.Camera) -> np.ndarray:
    """帧 -> 二值水马 mask（0/255，同尺寸）。"""
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    roi = rect_mask(frame.shape, cam.roi) & ~rect_mask(frame.shape, cam.exclude)
    far_band = rect_mask(frame.shape, cam.far_band) & roi
    k_open, k_close, k_knit_h, k_knit_v, min_cc, min_cc_far = kernels(*frame.shape[:2][::-1])
    fg = build_fg(hsv, lab, far_band, k_open, k_close, k_knit_h, k_knit_v) & roi
    return clean(fg, hsv, lab, far_band, min_cc, min_cc_far)
