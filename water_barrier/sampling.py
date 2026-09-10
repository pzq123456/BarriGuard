"""Barrier-axis sampling. E5 detection logic reused unchanged.

Trims the old sampler: gray/edge columns (g/e) and raw station counts were
only used by exploratory probes, never by the frozen gapwatch path.
Per-station y-extent (y0/y1) is measured geometry for pixel-box output;
it does not feed detection.
"""

import cv2
import numpy as np

from .detector import REF, red_m, white_m

STEP = 2.0
MINH = 10


def _row_axis(poly_norm):
    c = poly_norm.mean(axis=0)
    _, _, vt = np.linalg.svd(poly_norm - c, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    if d[0] < 0:
        d = -d
    rel = (poly_norm - c) @ d
    return {"d": d.astype(np.float32), "c": c.astype(np.float32),
            "umin": float(rel.min()), "umax": float(rel.max())}


def _pack(col, foreign, valid_station, m, umin, step, d, c):
    def rs(a, interp):
        return cv2.resize(a.reshape(1, -1), (REF, 1), interpolation=interp).ravel()

    v = rs(valid_station.astype(np.uint8), cv2.INTER_NEAREST) > 0
    v = cv2.erode(v.astype(np.uint8).reshape(1, -1), np.ones((1, 7))).ravel() > 0
    return {"s": rs(col.astype(np.float32), cv2.INTER_LINEAR),
            "f": rs(foreign.astype(np.float32), cv2.INTER_LINEAR),
            "v": v, "M": m, "umin": umin, "step": step, "d": d, "c": c}


def _resample_y(ym, cnt):
    """Per-M-station y-extent -> REF arrays. Gaps filled by index interp (measured)."""
    ym = ym.astype(np.float32)
    ym[cnt == 0] = np.nan
    idx = np.arange(len(ym))
    ok = np.isfinite(ym)
    if not ok.any():
        return np.full(REF, np.nan, np.float32)
    filled = np.interp(idx, idx[ok], ym[ok]).astype(np.float32)
    return cv2.resize(filled.reshape(1, -1), (REF, 1), interpolation=cv2.INTER_LINEAR).ravel()


def sample_row(bgr, poly_norm, step_px=STEP, minh=MINH):
    """Sample along barrier axis. Section split matches E5: top 60% white, bottom 40% red."""
    h, w = bgr.shape[:2]
    pts = poly_norm * np.array([w, h], dtype=np.float32)
    ax = _row_axis(poly_norm)
    d = ax["d"] * np.array([w, h], dtype=np.float32)
    d = d / np.linalg.norm(d)
    c = ax["c"] * np.array([w, h], dtype=np.float32)
    rel = (pts - c) @ d
    umin, umax = float(rel.min()), float(rel.max())
    m = max(1, int(np.ceil((umax - umin) / step_px)))
    x0, y0 = pts.min(axis=0).astype(int).clip(0)
    x1, y1 = pts.max(axis=0).astype(int)
    x1, y1 = min(w - 1, int(x1)), min(h - 1, int(y1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mask, [(pts - np.array([x0, y0])).astype(np.int32)], 1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        z = np.zeros(m, np.float32)
        pack = _pack(z, z, np.zeros(m, bool), m, umin, step_px, d, c)
        pack["y0"] = np.full(REF, np.nan, np.float32)
        pack["y1"] = np.full(REF, np.nan, np.float32)
        return pack

    col_id = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
    station = np.floor(((col_id - c) @ d - umin) / step_px).astype(int).clip(0, m - 1)

    rm = red_m(bgr)
    wm = white_m(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32)
    is_red_hue = (hue <= 12) | (hue >= 165)
    foreign = ((sat > 100) & (~is_red_hue)).astype(np.float32)

    red_px = rm[ys + y0, xs + x0].astype(np.float32)
    white_px = wm[ys + y0, xs + x0].astype(np.float32)
    foreign_px = foreign[ys + y0, xs + x0]
    rows_y = (ys + y0).astype(np.float32)

    order = np.lexsort((rows_y, station))
    ss = station[order]
    cnt = np.bincount(ss, minlength=m)
    pos = np.arange(len(ss)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    is_white = pos < (0.6 * cnt[ss]).astype(int)

    white_sum = np.bincount(ss[is_white], weights=white_px[order][is_white], minlength=m)
    white_cnt = np.bincount(ss[is_white], minlength=m)
    red_sum = np.bincount(ss[~is_white], weights=red_px[order][~is_white], minlength=m)
    red_cnt = np.bincount(ss[~is_white], minlength=m)
    foreign_sum = np.bincount(ss, weights=foreign_px[order], minlength=m)

    valid_station = cnt >= minh
    col = np.zeros(m, np.float32)
    foreign_col = np.zeros(m, np.float32)
    ok = valid_station & (white_cnt > 0) & (red_cnt > 0)
    col[ok] = (0.6 * red_sum[ok] / red_cnt[ok] + 0.4 * white_sum[ok] / white_cnt[ok]).astype(np.float32)
    foreign_col[valid_station] = (foreign_sum[valid_station] / cnt[valid_station]).astype(np.float32)

    abs_y = (ys + y0).astype(np.float32)
    ymin_m = np.full(m, np.inf, np.float32)
    ymax_m = np.full(m, -np.inf, np.float32)
    np.minimum.at(ymin_m, station, abs_y)
    np.maximum.at(ymax_m, station, abs_y)

    pack = _pack(col, foreign_col, valid_station, m, umin, step_px, d, c)
    pack["y0"] = _resample_y(ymin_m, cnt)
    pack["y1"] = _resample_y(ymax_m, cnt)
    return pack


def apply_end_trim(pack, frac):
    """Mark both axis ends invalid: section collapses at polygon vertices."""
    v = pack["v"].copy()
    k = int(REF * frac)
    v[:k] = False
    v[REF - k:] = False
    v = cv2.erode(v.astype(np.uint8).reshape(1, -1), np.ones((1, 7))).ravel() > 0
    return {**pack, "v": v}


def sta_to_xy(pack, sta812, w, h):
    """Station (812 coords) -> image point, for overlay only."""
    s = (np.asarray(sta812, dtype=np.float32) + 0.5) / REF * pack["M"]
    pts = pack["c"] + np.outer(pack["umin"] + (s + 0.5) * pack["step"], pack["d"])
    pts[:, 0] = pts[:, 0].clip(0, w - 1)
    pts[:, 1] = pts[:, 1].clip(0, h - 1)
    return pts
