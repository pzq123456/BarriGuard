"""粗糙度/结构特征实验（问题 3 方向）：纹理粗糙度，而非条纹周期。

背景：自相关峰全落在 lag 下界（无条纹周期峰），真正分开路面/水马的是
“粗糙度/持续性”（路面颗粒 0.13~0.16 vs 水马区 0.4~0.8）。本模块提供可复用的
按站原始序列 + 峰突出度统计，供离线验证分离度。
注意：只分“水马材质 vs 路面”，分不开同材质的阴影/日光（见 R2/R3 报告）。

用法：python -m water_barrier.roughness --img data/1750/clean_01.jpg --row 0
"""
import argparse

import cv2 as cv
import numpy as np

from .detector import REF
from .sampling import STEP, _row_axis, sample_row


def autocorr(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    if float(x @ x) <= 1e-9:
        return np.zeros(len(x)), 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    return ac / ac[0], float(x.var())


def prominence(series, lo, hi):
    """lag 带内自相关峰突出度（max-median）。平坦序列返回 0。"""
    ac, var = autocorr(series)
    hi = min(hi, len(ac) - 1)
    band = ac[lo:hi + 1]
    return float(band.max() - np.median(band)), var, int(lo + np.argmax(band))


def station_series(bgr, poly):
    """行多边形 -> 每站原始序列 dict（s 未平滑 + 均值 V/S，重采样到 812 站）。

    关键：用未平滑像素算粗糙度——高斯平滑会把颗粒纹理磨掉（R2 实验踩过的坑）。
    """
    h, w = bgr.shape[:2]
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    pack = sample_row(bgr, np.asarray(poly, np.float32))
    pts = np.asarray(poly, np.float32) * np.array([w, h], np.float32)
    ax = _row_axis(np.asarray(poly, np.float32))
    d = ax["d"] * np.array([w, h], np.float32)
    d = d / np.linalg.norm(d)
    c = ax["c"] * np.array([w, h], np.float32)
    rel = (pts - c) @ d
    umin = float(rel.min())
    m = max(1, int(np.ceil((float(rel.max()) - umin) / STEP)))
    x0, y0 = pts.min(axis=0).astype(int).clip(0)
    xe, ye = min(w - 1, int(pts.max(axis=0)[0])), min(h - 1, int(pts.max(axis=0)[1]))
    mask = np.zeros((ye - y0 + 1, xe - x0 + 1), np.uint8)
    cv.fillPoly(mask, [(pts - np.array([x0, y0])).astype(np.int32)], 1)
    ys, xs = np.where(mask > 0)
    st = np.floor((((np.stack([xs + x0, ys + y0], 1)).astype(np.float32) - c)
                   @ d - umin) / STEP).astype(int).clip(0, m - 1)
    cnt = np.maximum(np.bincount(st, minlength=m), 1)
    out = {}
    out["s_raw"] = pack["s"]
    for name, ch in (("V", hsv[:, :, 2]), ("S", hsv[:, :, 1])):
        avg = np.bincount(st, weights=ch[ys + y0, xs + x0], minlength=m) / cnt
        out[name] = cv.resize(avg.reshape(1, -1), (REF, 1),
                              interpolation=cv.INTER_LINEAR).ravel()
    return out


def main():
    from server.config import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="1750")
    ap.add_argument("--img", default="data/1750/clean_01.jpg")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--lo", type=int, default=15)
    ap.add_argument("--hi", type=int, default=90)
    ap.add_argument("--wins", default="200:360,363:523,40:200",
                    help="站区间逗号分隔（默认：完好红/阴影FP/近端混合）")
    args = ap.parse_args()
    params = load()
    calib = next(c for c in params["cameras"] if c["id"] == args.camera)["algos"]["water_gap"]
    bgr = cv.imread(args.img)
    series = station_series(bgr, calib["rows"][args.row]["poly"])
    for wdef in args.wins.split(","):
        a, b = (int(x) for x in wdef.split(":"))
        for sn in ("s_raw", "V", "S"):
            p, var, lag = prominence(series[sn][a:b], args.lo, args.hi)
            print(f"[{a}:{b}] {sn:5s} 突出度={p:.3f} 方差={var:.1f} 峰lag={lag}")


if __name__ == "__main__":
    main()
