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
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 1] <= 70) & (hsv[:, :, 2] >= 150)


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


def detect(sm, bs, v):
    """Envelope discovery d>0.25, extend d>0.125; core is S<FLOOR_MAX run."""
    d = bs - sm
    outs, i, n = [], 0, len(sm)
    while i < n:
        if (not v[i]) or d[i] <= 0.25:
            i += 1
            continue
        j = i
        while j < n and v[j] and d[j] > 0.125:
            j += 1
        if v[i:j].sum() / max(1, j - i) < 0.9 or (j - i) < MIN_WIDTH:
            i = j
            continue
        env = (int(i), int(j - 1))
        k, best = i, None
        while k < j:
            if sm[k] >= FLOOR_MAX:
                k += 1
                continue
            m = k
            while m < j and sm[m] < FLOOR_MAX + CORE_EXIT:
                m += 1
            e = m
            while e - 1 >= k and sm[e - 1] >= FLOOR_MAX:
                e -= 1
            if e - k >= CORE_MIN_WIDTH:
                best = (int(k), int(e - 1), int(e - k), round(float(sm[k:e].min()), 3))
                break
            k = m
        outs.append({"env": env, "core": best, "prom": round(float(d[i:j].max()), 3),
                     "reason": "FLOOR+WIDTH" if best else "FLOOR_FAIL"})
        i = j
    return outs
