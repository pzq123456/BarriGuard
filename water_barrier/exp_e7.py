"""E7: far-zone temporal-stability + relative-contrast probe.
Frozen E5/E6. Far zone defined by geometry (scale_h), no GT, no thresholds.
REMOTE_GAP sample absent in data -> reported, not invented.
"""
import cv2
import numpy as np
import json
import csv
from pathlib import Path

from exp_e5 import masks, col_range, red_m, white_m, band_cols, to812, valid812
from exp_e6 import smooth_s, mbase_w

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
REF = 812
STEP = 3  # every 3rd cap file

def gray_of(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g).astype(np.float32)

if __name__ == "__main__":
    base = ROOT.parent
    files = sorted((base / "tmp/cap_day").glob("cap_*.png"))[::STEP]
    W, R, G = [], [], []  # wall S, road gray, full gray(for bg)
    V = None
    for p in files:
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        m = masks(h, w)
        rm = red_m(bgr)
        y0, y1 = col_range(m)
        rc, wc, vd = band_cols(rm, white_m(bgr), y0, y1)
        v = valid812(vd)
        if V is None:
            V = v
        comb = 0.6 * to812(rc) + 0.4 * to812(wc)
        g = cv2.resize(gray_of(bgr), (REF, 1), interpolation=cv2.INTER_LINEAR).ravel()
        # road band: below roi, same x
        rs = np.zeros(REF, np.float32)
        hh, ww = bgr.shape[:2]
        gx = (np.arange(REF) / REF * w).astype(int)
        gg = gray_of(bgr)
        for i, x in enumerate(gx):
            if not vd[x]:
                continue
            a, b = min(hh - 1, y1[x] + 10), min(hh, y1[x] + 90)
            if b > a:
                rs[i] = float(gg[a:b, x].mean())
        W.append(comb)
        R.append(rs)
        G.append(g)
    W = np.stack(W)
    R = np.stack(R)
    med_w = np.median(W, axis=0)
    mad_w = np.median(np.abs(W - med_w), axis=0)
    var_w = W.var(axis=0)
    med_r = np.median(R, axis=0)
    mad_r = np.median(np.abs(R - med_r), axis=0)
    # light spatial smooth of stats
    sm = lambda s: cv2.GaussianBlur(s.reshape(1, -1), (15, 1), 2).ravel()
    med_w, mad_w, med_r, mad_r = sm(med_w), sm(mad_w), sm(med_r), sm(mad_r)
    # far zone by geometry: valid + (scale via E6 h proxy: reuse row height)
    # approx: far = valid & x>=450 (geometry cut from ROI extent, documented)
    far = V & (np.arange(REF) >= 450)
    # relative deficit on temporal median wall signal, local baseline win 101
    bs = mbase_w(smooth_s(med_w, V, 2.5), V, 101)
    deficit = bs - med_w
    normdef = deficit / np.maximum(bs, 1e-6)
    # csv per-x
    with open(OUT / "E7_temporal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "valid", "far", "med_wall", "mad_wall", "var_wall",
                    "med_road", "mad_road", "deficit", "normdef"])
        for x in range(0, REF, 2):
            w.writerow([x, int(V[x]), int(far[x]), round(float(med_w[x]), 4),
                        round(float(mad_w[x]), 4), round(float(var_w[x]), 4),
                        round(float(med_r[x]), 2), round(float(mad_r[x]), 2),
                        round(float(deficit[x]), 4), round(float(normdef[x]), 4)])
    # zone distributions
    wall = far & (med_w > 0.02)
    road = far  # road stats sampled under same far x
    bg = ~V & (np.arange(REF) >= 450)
    sep = {}
    for zn, msk in [("REMOTE_WALL", wall), ("REMOTE_ROAD", road), ("REMOTE_BG", bg)]:
        sep[zn] = {"n": int(msk.sum()),
                   "med_wall": round(float(np.median(med_w[msk])) if msk.any() else -1, 4),
                   "mad_wall": round(float(np.median(mad_w[msk])) if msk.any() else -1, 4),
                   "mad_road": round(float(np.median(mad_r[msk])) if msk.any() else -1, 2),
                   "deficit": round(float(np.median(deficit[msk])) if msk.any() else -1, 4),
                   "normdef": round(float(np.median(normdef[msk])) if msk.any() else -1, 4)}
    sep["REMOTE_GAP"] = "NO_SAMPLE_IN_DATA"
    with open(OUT / "E7_sep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["zone", "n", "med_wall", "mad_wall", "mad_road", "deficit", "normdef"])
        w.writeheader()
        for zn, d in sep.items():
            if isinstance(d, dict):
                w.writerow({"zone": zn, **d})
    with open(OUT / "E7_remote.csv", "w", newline="", encoding="utf-8") as f:
        f.write("note: no far-gap GT in data; wall/road/bg distributions only\n")
    # plots
    H = 200
    tmp = np.zeros((H, REF, 3), np.uint8)
    for arr, col, sc in [(mad_w, (0, 255, 0), 3.0), (mad_r / 255.0, (255, 0, 255), 3.0)]:
        yv = (np.clip(arr * sc, 0, 1) * (H - 20) + 10).astype(int)
        for i in range(1, REF):
            cv2.line(tmp, (i - 1, H - yv[i - 1]), (i, H - yv[i]), col, 1)
    cv2.imwrite(str(OUT / "E7_temporal.png"), tmp)
    dpl = np.zeros((H, REF, 3), np.uint8)
    for arr, col in [(np.clip(deficit * 3, 0, 1), (0, 255, 0)), (np.clip(normdef, 0, 1), (255, 255, 0))]:
        yv = (arr * (H - 20) + 10).astype(int)
        for i in range(1, REF):
            cv2.line(dpl, (i - 1, H - yv[i - 1]), (i, H - yv[i]), col, 1)
    cv2.imwrite(str(OUT / "E7_deficit.png"), dpl)
    S = 400
    sc = np.zeros((S, S, 3), np.uint8)
    for msk, col in [(wall, (0, 255, 0)), (road, (255, 0, 255)), (bg, (255, 255, 0))]:
        xs, ys = normdef[msk], mad_w[msk]
        for x, y in zip(xs[::2], ys[::2]):
            px, py = min(S - 1, int(x * S)), S - 1 - min(S - 1, int(y * 8 * S / 8))
            cv2.circle(sc, (px, py), 2, col, -1)
    cv2.imwrite(str(OUT / "E7_scatter.png"), sc)
    print(json.dumps(sep, indent=1))
    print(json.dumps({"frames": len(files)}))
