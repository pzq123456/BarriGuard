"""盲测：零灯位先验，全帧只凭闪烁定位，前后半段独立验证.

代码里没有任何灯坐标、没有任何分带ROI：全帧逐像素max/min/mean/std ->
闪烁掩膜 -> NMS -> 逐候选亮度序列 -> 前半/后半各自判周期 -> 两套点集互匹配.
对得上=位置来自数据而非先验. 输出 output/blind.json, output/blind_map.png
"""
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "output"
VID = Path(r"C:\Users\admin\Desktop\work\BarriGuard\tmp\1749_202609040100.mp4")

STEP = 3
RNG_TH, STD_TH, MAX_TH, MEAN_MAX = 80, 16.0, 150, 170
MIN_AREA, NMS_R = 8, 18
LAG_LO, LAG_HI, AC_TH = 3, 9, 0.45
MEAN_SWING, MAX_SWING = 40.0, 70.0
MATCH_R = 15.0


def full_stats():
    mx = mn = ssum = ssum2 = None
    n = 0
    fi = 0
    cap = cv.VideoCapture(str(VID))
    while True:
        ok = cap.grab()
        if not ok:
            break
        if fi % STEP == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY).astype(np.float32)
            if mx is None:
                mx, mn, ssum, ssum2 = g.copy(), g.copy(), g.copy(), g * g
            else:
                np.maximum(mx, g, out=mx)
                np.minimum(mn, g, out=mn)
                ssum += g
                ssum2 += g * g
            n += 1
        fi += 1
    cap.release()
    mean = ssum / n
    std = np.sqrt(np.maximum(ssum2 / n - mean ** 2, 0))
    return mx, mn, mean, std, mx - mn, n


def propose(mx, mn, mean, std, rng):
    m = ((rng >= RNG_TH) & (std >= STD_TH) & (mx >= MAX_TH) & (mean <= MEAN_MAX))
    m = cv.morphologyEx(m.astype(np.uint8) * 255, cv.MORPH_OPEN,
                        np.ones((3, 3), np.uint8))
    ncc, _, stats, cents = cv.connectedComponentsWithStats(m, 8)
    raw = [(float(cents[i][0]), float(cents[i][1]), int(stats[i, 4]))
           for i in range(1, ncc) if stats[i, 4] >= MIN_AREA]
    raw.sort(key=lambda t: -t[2])
    sel = []
    for x, y, _ in raw:
        if all((x - sx) ** 2 + (y - sy) ** 2 >= NMS_R ** 2 for sx, sy in sel):
            sel.append((x, y))
    print(f"raw={len(raw)} nms={len(sel)}", flush=True)
    return sel


def series_xy(pts):
    Sx = [[] for _ in pts]
    Sm = [[] for _ in pts]
    cap = cv.VideoCapture(str(VID))
    fi = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if fi % STEP == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
            for i, (x, y) in enumerate(pts):
                xi, yi = int(round(x)), int(round(y))
                roi = g[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4]
                Sx[i].append(float(roi.max()))
                Sm[i].append(float(roi.mean()))
        fi += 1
    cap.release()
    return np.array(Sx), np.array(Sm)


def is_flash(vx, vm):
    def per(v):
        v = np.asarray(v, float)
        p10, p90 = float(np.percentile(v, 10)), float(np.percentile(v, 90))
        on = v >= p10 + 0.4 * (p90 - p10)
        d = float(on.mean())
        x = on.astype(float) - on.mean()
        if float(x @ x) <= 0:
            return d, 0.0, 0
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        ac /= ac[0]
        seg = ac[LAG_LO:LAG_HI + 1]
        return d, float(seg.max()), int(np.argmax(seg)) + LAG_LO
    dm, acm, lagm = per(vm)
    dx, acx, lagx = per(vx)
    d, ac, lag = (dm, acm, lagm) if acm >= acx else (dx, acx, lagx)
    msw = float(np.percentile(vm, 90) - np.percentile(vm, 10))
    xsw = float(np.percentile(vx, 90) - np.percentile(vx, 10))
    ok = (ac >= AC_TH and 0.06 <= d <= 0.70
          and (msw >= MEAN_SWING or xsw >= MAX_SWING)
          and float(np.percentile(vx, 90)) >= 180)
    return ok, round(d, 3), round(ac, 2), lag


def match(A, B):
    used, pairs = set(), []
    for i, a in enumerate(A):
        best, bd = None, MATCH_R ** 2
        for j, b in enumerate(B):
            if j in used:
                continue
            dd = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if dd < bd:
                best, bd = j, dd
        if best is not None:
            used.add(best)
            pairs.append((i, best, round(float(np.sqrt(bd)), 1)))
    return pairs


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    mx, mn, mean, std, rng, n = full_stats()
    print(f"samples={n} frame={mx.shape}", flush=True)
    pts = propose(mx, mn, mean, std, rng)
    Sx, Sm = series_xy(pts)
    h = Sx.shape[1] // 2
    A, B, detA, detB, Aidx = [], [], {}, {}, []
    for i, (x, y) in enumerate(pts):
        for tag, vx, vm, out in (("A", Sx[i, :h], Sm[i, :h], A),
                                 ("B", Sx[i, h:], Sm[i, h:], B)):
            ok, d, ac, lag = is_flash(vx, vm)
            if ok:
                out.append((round(x, 1), round(y, 1)))
                if tag == "A":
                    Aidx.append(i)
                (detA if tag == "A" else detB)[f"{round(x, 1)},{round(y, 1)}"] = \
                    dict(duty=d, ac=ac, lag=lag)
    pairs = match(A, B)
    onlyA = len(A) - len(pairs)
    onlyB = len(B) - len(pairs)
    print(f"A={len(A)} B={len(B)} matched={len(pairs)} onlyA={onlyA} onlyB={onlyB}",
          flush=True)
    for i, j, dd in pairs:
        print(f"  A{A[i]} <-> B{B[j]} d={dd}", flush=True)
    for tag, arr in (("Aonly", [A[i] for i in range(len(A))
                                if i not in {p[0] for p in pairs}]),
                     ("Bonly", [B[j] for j in range(len(B))
                                if j not in {p[1] for p in pairs}])):
        for p in arr:
            print(f"  {tag} {p}", flush=True)
    json.dump(dict(A=A, B=B, pairs=pairs, detA=detA, detB=detB,
                   agree=round(len(pairs) / max(1, max(len(A), len(B))), 3)),
              open(OUT / "blind.json", "w", encoding="utf-8"), ensure_ascii=False)
    cap = cv.VideoCapture(str(VID))
    cap.set(cv.CAP_PROP_POS_FRAMES, int(cap.get(cv.CAP_PROP_FRAME_COUNT)) // 2)
    _, ref = cap.read()
    cap.release()
    ma = {p[0] for p in pairs}
    mb = {p[1] for p in pairs}
    for i, p in enumerate(A):
        col = (0, 255, 0) if i in ma else (0, 165, 255)
        cv.circle(ref, (int(p[0]), int(p[1])), 9, col, 2)
    for j, p in enumerate(B):
        if j not in mb:
            cv.drawMarker(ref, (int(p[0]), int(p[1])), (255, 0, 0),
                          cv.MARKER_DIAMOND, 16, 2)
    cv.imwrite(str(OUT / "blind_map.png"), ref)
    # matched spots brightness curves (whole-video series)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(pairs)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), sharex=True)
        for ax, p in zip(np.asarray(axes).flat, pairs):
            v = Sm[Aidx[p[0]]][::10]
            ax.plot(v, linewidth=0.5)
            ax.set_title(f"A{A[p[0]]}", fontsize=8, color="g")
        fig.suptitle("matched flash spots brightness (7x7 mean, /10)")
        fig.tight_layout()
        fig.savefig(OUT / "spot_curves.png", dpi=120)
    except ImportError:
        pass
    print(f"done {time.time()-t0:.0f}s -> output/blind.json", flush=True)


if __name__ == "__main__":
    main()
