"""分割基座：把"水马"（红底座 + 白板）从白天实流里稳健地割出来。

相比第一版纯"红∪白"色阈值的改进：
  1. 红底座锚定：红色（色相红 + S + a*）几乎不会出现在路面/树影/灰车上，是最稳的信号。
     先用红底座定位水马，再只保留"靠近红底座的白板"，从而弃掉远处高亮但不属于水马的
     白色路面、白色货车、天空等地物。
  2. 近严远宽：远端带色相不稳、S 低，放宽阈值，但仍以红底座锚定。
  3. 形态学：开运算去噪 + 闭运算 + 沿路向线段核把断缝抹平，连成屏障带。
  4. 连通域稳定化（'largest + second-largest' 思想）：只保留面积前几大连通块（含红/白校验）。

阈值来源：data 已标定版本（1622x905）；均为绝对像素，无分辨率缩放，多相机应各自标定。
所有阈值/区域均为可调参数，经 SegmentCfg 注入；默认值为标定值。
"""
from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class SegmentCfg:
    """水马分割的可调参数（绝对像素单位；无分辨率缩放）。"""
    # 色彩阈值
    red_h: tuple = (0, 10, 168, 180)
    red_s_min: int = 60
    red_a_min: int = 145
    white_v_min: int = 195
    white_s_max: int = 60
    far_red_s_min: int = 25
    far_red_a_min: int = 134
    far_red_v_min: int = 140
    far_white_v_min: int = 145   # 远端白板更暗；红底座锚定剔除非设施白色，故放宽 V 安全
    far_white_s_max: int = 80
    # 连通域均值色体检 + 稳定化
    cc_a_redish: int = 136
    cc_v_whiteish: int = 145
    cc_s_whiteish: int = 80
    max_keep: int = 3
    # 形态学核（绝对像素）
    open_k: tuple = (3, 3)
    close_k: tuple = (9, 9)
    knit_h: tuple = (1, 21)
    knit_v: tuple = (21, 1)
    anchor_k: tuple = (9, 9)
    min_cc_area: int = 12
    far_min_cc_area: int = 6
    # 几何区域（绝对像素）
    roi: tuple = ((0, 330, 900, 930), (1040, 295, 1462, 395), (880, 370, 1468, 950))
    far_band: tuple = ((1040, 295, 1462, 395), (1040, 375, 1170, 445))
    exclude: tuple = ((1100, 25, 1595, 135), (15, 825, 125, 900),
                      (1272, 280, 1318, 306), (1420, 272, 1458, 306))


def _rect_mask(shape, rects, val=255):
    m = np.zeros(shape[:2], np.uint8)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = val
    return m


def _color_masks(hsv, lab, cfg, s_red, a_red, v_red, v_white, s_white, hue):
    """一组（红,白）掩膜；远端版多一个 V 下限防暗色噪声。hue 传入复用。"""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = hue & cv.inRange(s, s_red, 255) & cv.inRange(lab[..., 1], a_red, 255)
    if v_red:
        red &= cv.inRange(v, v_red, 255)
    white = cv.inRange(v, v_white, 255) & cv.inRange(s, 0, s_white)
    return red, white


def _cc_mean(hsv, lab, labels, i, x, y, w, h):
    m = labels[y:y + h, x:x + w] == i
    return (lab[..., 1][y:y + h, x:x + w][m].mean(),
            hsv[..., 2][y:y + h, x:x + w][m].mean(),
            hsv[..., 1][y:y + h, x:x + w][m].mean())  # a*, V, S


def _keep_big(fg, far_band, min_cc, min_cc_far, hsv, lab, cfg):
    """连通域清杂 + 稳定化：剔除小斑、颜色既非红也非白的、以及面积掉出前 max_keep 的。"""
    n, labels, stats, _ = cv.connectedComponentsWithStats(fg, 8)

    kept_idx = []
    for i in range(1, n):
        x, y, w, h, area = stats[i, :5]
        if area < (min_cc_far if far_band[y + h // 2, x + w // 2] else min_cc):
            continue
        a, v, s = _cc_mean(hsv, lab, labels, i, x, y, w, h)
        if a >= cfg.cc_a_redish or (v >= cfg.cc_v_whiteish and s <= cfg.cc_s_whiteish):
            kept_idx.append(i)

    if not kept_idx:
        return np.zeros_like(fg)
    kept_idx.sort(key=lambda i: stats[i, cv.CC_STAT_AREA], reverse=True)
    kept_idx = kept_idx[:cfg.max_keep]

    keep = np.zeros(n, bool)
    keep[kept_idx] = True
    return keep[labels].astype(np.uint8) * 255


def _white_near_red(white, red, gap):
    """保留"紧贴红底座"的白板连通块：白板与（膨胀小间隙后的）红底座相交即整体保留。

    向量化：对每个 CC 号，用 bincount 统计它覆盖的膨胀红像素数，>=1 即保留该 CC。
    避免逐 CC 做整帧 labels==i 布尔重建（曾是 1920x1080 下最大的耗时点）。
    """
    red_d = cv.dilate(red, cv.getStructuringElement(cv.MORPH_RECT, gap))
    n, labels = cv.connectedComponents(white, 8)
    hit = np.bincount(labels[red_d > 0].ravel(), minlength=n)
    keep = hit > 0
    keep[0] = False  # label 0 是背景，排除
    return (keep[labels].astype(np.uint8)) * 255


def segment(frame: np.ndarray, cfg: SegmentCfg = None) -> np.ndarray:
    """帧 -> 稳健水马 mask（0/255，同尺寸）。cfg 为 None 时用默认标定值。"""
    cfg = cfg or SegmentCfg()
    h, w = frame.shape[:2]
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    roi = _rect_mask((h, w), cfg.roi)
    far_band = _rect_mask((h, w), cfg.far_band) & roi

    k_open, k_close, k_knit_h, k_knit_v, k_anchor = (cfg.open_k, cfg.close_k,
                                                     cfg.knit_h, cfg.knit_v, cfg.anchor_k)
    min_cc, min_cc_far = cfg.min_cc_area, cfg.far_min_cc_area

    hh = hsv[..., 0]
    hue = cv.inRange(hh, cfg.red_h[0], cfg.red_h[1]) | cv.inRange(hh, cfg.red_h[2], cfg.red_h[3])
    red, white = _color_masks(hsv, lab, cfg, cfg.red_s_min, cfg.red_a_min, 0,
                              cfg.white_v_min, cfg.white_s_max, hue)
    red_f, white_f = _color_masks(hsv, lab, cfg, cfg.far_red_s_min, cfg.far_red_a_min,
                                  cfg.far_red_v_min, cfg.white_v_min, cfg.white_s_max, hue)
    far_white = cv.inRange(hsv[..., 2], cfg.far_white_v_min, 255) & \
        cv.inRange(hsv[..., 1], 0, cfg.far_white_s_max)
    for m in (red, white, red_f, white_f, far_white):
        m[:] = cv.morphologyEx(m, cv.MORPH_OPEN, np.ones(k_open, np.uint8))

    white = _white_near_red(white, red | red_f, k_anchor)
    white_f = _white_near_red(white_f, red | red_f, k_anchor)
    far_white = _white_near_red(far_white, red | red_f, k_anchor)

    fg = (red | white) | ((red_f | white_f | far_white) & far_band)

    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(k_close, np.uint8))
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(k_knit_h, np.uint8))
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(k_knit_v, np.uint8))
    fg &= (~_rect_mask((h, w), cfg.exclude)) & roi

    return _keep_big(fg, far_band, min_cc, min_cc_far, hsv, lab, cfg)
