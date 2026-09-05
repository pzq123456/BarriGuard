"""自主缺口检测：在"最大/次大连通域"稳定化的分割基座上定位缺口。

关键认知：缺口大到一定程度会把整条水马线"切断"，变成两三个大连通块。因此缺口有两类：
  1. 块间断口（inter）：同一水马线上的相邻两块之间出现开口（真正的水马被移走）。
  2. 块内缺口（intra）：单个大块内部覆盖率骤降（小块/遮挡/局部缺失）。

做法：
  A. 取分割基座面积前若干的大连通块（含红/白校验），丢弃零星噪声块；
  B. 块间断口：把大块按水平先后排序，相邻两块若纵向重叠较大（同一条线）且横向开口
     落在 [min_gap, max_gap] 内 -> 报为移除缺口；开口过大说明是行边界/场景尺度，忽略。
  C. 块内缺口：每块内再做逐列带覆盖率，找覆盖 < thr 的内部低覆盖段。

仅有的先验是场景粗尺度（连通块大小 / 缺口宽度窗口），不含"缺口在哪"。
"""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

TOP_PIECES = 6
MIN_GAP_PX = 30.0


@dataclass
class Piece:
    mask: np.ndarray
    x0: int
    x1: int
    y0: int
    y1: int


@dataclass
class Gap:
    x0: int
    y0: int
    x1: int
    y1: int
    severity: float
    kind: str  # "inter" / "intra"


def _fit(mask):
    ys, xs = np.nonzero(mask)
    return Piece(mask, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))


def _pieces(fg, min_area, top):
    n, labels, stats, _ = cv.connectedComponentsWithStats(fg, 8)
    cand = [labels == i for i in range(1, n)
            if stats[i, cv.CC_STAT_AREA] >= min_area]
    cand.sort(key=lambda m: cv.countNonZero(m), reverse=True)
    return [_fit(m) for m in cand[:top]]


def _intra_cov(fg, piece, bin_col=8):
    """块内逐列带覆盖率：带高随透视自动收缩。"""
    W = piece.x1 - piece.x0 + 1
    if W < 20:
        return None
    nb = max(8, int(round(W / bin_col)))
    top, bot = np.full(nb, np.nan), np.full(nb, np.nan)
    for b in range(nb):
        l = piece.x0 + int(b * W / nb); r = piece.x0 + int((b + 1) * W / nb)
        m = piece.mask[piece.y0:piece.y1, l:r]
        ys = m.nonzero()[0]
        if len(ys) >= 6:
            top[b] = ys.min() + piece.y0; bot[b] = ys.max() + piece.y0
    valid = np.isfinite(top)
    if valid.sum() < 3:
        return None
    top = np.interp(np.arange(nb), np.flatnonzero(valid), top[valid])
    bot = np.interp(np.arange(nb), np.flatnonzero(valid), bot[valid])
    cov = np.zeros(nb)
    for b in range(nb):
        l = piece.x0 + int(b * W / nb); r = piece.x0 + int((b + 1) * W / nb)
        y0i, y1i = int(max(0, top[b])), int(min(fg.shape[0], bot[b] + 1))
        win = fg[y0i:y1i, l:r]
        cov[b] = min(1.0, cv.countNonZero(win) / float(win.size))
    return cov, nb, top, bot


def detect_gaps(fg, min_area=3000, thr=0.5, min_len=2, end_margin=0.04,
                min_gap_px=MIN_GAP_PX, max_gap_px=None):
    """返回 Gap 列表（bbox + 严重度 + 类型），无缺口位置预标记。

    min_gap_px / max_gap_px：块间断口仅当开口落在 [min, max] 内才报警。
    max（默认=min*3）：防止把"两段本是同一向纵深延伸的墙"或"跨行"误判成缺口。
    """
    if max_gap_px is None:
        max_gap_px = min_gap_px * 3
    pieces = _pieces(fg, min_area, TOP_PIECES)
    gaps = []

    # 1) 块间断口
    ps = sorted(pieces, key=lambda p: p.x0)
    for a, b in zip(ps, ps[1:]):
        opening = b.x0 - a.x1
        if not (min_gap_px <= opening <= max_gap_px):
            continue
        inter_h = min(a.y1, b.y1) - max(a.y0, b.y0)
        min_h = min(a.y1 - a.y0, b.y1 - b.y0)
        if min_h <= 0 or inter_h < 0.5 * min_h:
            continue  # 纵向基本不重叠 -> 不是同一条线的相邻两块
        mx = (a.x1 + b.x0) // 2
        my = (a.y0 + a.y1) // 2
        hw = int(opening // 2)
        gaps.append(Gap(mx - hw, my - hw, mx + hw, my + hw,
                        min(1.0, float(opening) / (min_gap_px * 2)), "inter"))

    # 2) 块内低覆盖
    for p in ps:
        res = _intra_cov(fg, p)
        if res is None:
            continue
        cov, nb, top, bot = res
        m = max(1, int(round(nb * end_margin)))
        i = m
        while i < nb - m:
            if cov[i] < thr:
                j = i
                while j < nb - m and cov[j] < thr:
                    j += 1
                if j - i >= min_len:
                    spawn = (p.x1 - p.x0 + 1) / nb
                    gx0 = int(p.x0 + i * spawn); gx1 = int(p.x0 + (j + 1) * spawn)
                    gy0 = int(max(0, top[i])); gy1 = int(min(fg.shape[0], bot[j] + 1))
                    gaps.append(Gap(gx0, gy0, gx1, gy1,
                                    float(cov[i:j + 1].mean()), "intra"))
                i = j
            else:
                i += 1
    return gaps
