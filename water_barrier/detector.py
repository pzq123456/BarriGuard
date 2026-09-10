"""E5 frozen detector: color signal smoothing + envelope/core detection.

Only the code path used by the Round6 mainline is kept.
Multi-scale (E6) and far-zone temporal (E7) probes are dropped: no importer.
Image-x sampling helpers (masks/col_range/band_cols/to812/valid812) and the
cap_day temporal main are dropped: mainline samples along the barrier axis.
"""

import cv2
import numpy as np

REF = 812

FLOOR_MAX = 0.18
CORE_EXIT = 0.06
MIN_WIDTH = 40
CORE_MIN_WIDTH = 30
SMOOTH_SIGMA = 6
BASE_WIN = 251


def red_m(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    r = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    r = r | cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    return r > 0


def white_m(bgr):
    """水马白（含阴影白）：日光白 S<=70&V>=150；阴影白 S 70~120&V 80~150&H>=85。

    阴影白依据 2026-09-10 实测：阴影水马 HSV~(90,80,110)，与晴天路面 (90,20,170)
    在 S 上差 4 倍（天空光偏冷 vs 日光直射）。H>=85 剔绿叶（H~60~75），路面 H 无意义
    （低 S）故不受影响。阴影白计入 white 后采样信号与 RER 前景同步生效。
    残留风险：红棕泥土（H 小）仍会落入；S>120 的深影白仍走旧逻辑。回归矩阵看守。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sunlit = (s <= 70) & (v >= 150)
    shadow = (s > 70) & (s <= 120) & (v >= 80) & (v < 150) & (h >= 85)
    return sunlit | shadow


def smooth(s, v):
    ss = s.copy()
    if (~v).any() and v.any():
        idx = np.arange(len(s))
        ss[~v] = np.interp(idx[~v], idx[v], s[v])
    k = SMOOTH_SIGMA * 6 + 1
    return cv2.GaussianBlur(ss.reshape(1, -1), (k, 1), SMOOTH_SIGMA).ravel()


def _masked_mean(sm, v):
    num = cv2.blur((sm * v).reshape(1, -1), (BASE_WIN, 1)).ravel()
    den = cv2.blur(v.astype(np.float32).reshape(1, -1), (BASE_WIN, 1)).ravel()
    return num / np.maximum(den, 1e-6)


def mbase(sm, v):
    mean = _masked_mean(sm, v)
    tmp = sm.copy().astype(np.float32)
    tmp[~v] = -np.inf
    with np.errstate(invalid="ignore"):
        mx = cv2.dilate(tmp.reshape(1, -1), np.ones((1, BASE_WIN))).ravel()
    mx[~v] = np.nan
    return np.maximum(np.nan_to_num(mean), np.nan_to_num(mx))


def detect(sm, bs, v, floor_max=FLOOR_MAX, core_exit=CORE_EXIT,
           min_width=MIN_WIDTH, core_min_width=CORE_MIN_WIDTH):
    """Envelope discovery d>0.25, extend d>0.125; core is S<floor_max run."""
    d = bs - sm
    outs, i, n = [], 0, len(sm)
    while i < n:
        if (not v[i]) or d[i] <= 0.25:
            i += 1
            continue
        j = i
        while j < n and v[j] and d[j] > 0.125:
            j += 1
        if v[i:j].sum() / max(1, j - i) < 0.9 or (j - i) < min_width:
            i = j
            continue
        env = (int(i), int(j - 1))
        k, best = i, None
        while k < j:
            if sm[k] >= floor_max:
                k += 1
                continue
            m = k
            while m < j and sm[m] < floor_max + core_exit:
                m += 1
            e = m
            while e - 1 >= k and sm[e - 1] >= floor_max:
                e -= 1
            if e - k >= core_min_width:
                best = (int(k), int(e - 1), int(e - k), round(float(sm[k:e].min()), 3))
                break
            k = m
        outs.append({"env": env, "core": best, "prom": round(float(d[i:j].max()), 3),
                     "reason": "FLOOR+WIDTH" if best else "FLOOR_FAIL"})
        i = j
    return outs
