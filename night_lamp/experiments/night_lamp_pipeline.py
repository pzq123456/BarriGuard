"""夜间警示灯熄灭检测 v3 (final research pipeline).

单遍扫描, 逐窗口(3min):
  - 半分辨率逐像素 16-bin 直方图 → 逐像素 p50 背景(近邻上采样)
  - 全窗口逐像素 max → flash = max - bg → 连通域 → spot
  - 琥珀亮mask累计(常亮灯), 灯罩mask累计(静态灯罩)
  - 已知 spot 缓存 20x20 ROI patch → 窗口末以本窗口 bg 计算闪光序列
分类:
  ALIVE-FLASH  : duty∈[5,85]%, 自相关 lag∈[5,12] ac>=0.35, on窗口占比>=60%
  ALIVE-STEADY : 静态高亮琥珀(常亮灯)
  DEAD         : 静态灯罩 且 18px(dy<=12)内无 ALIVE 且 非(常亮灯正下方<=55px 反射) 且 y<1065
"""
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VID = Path(r"C:\Users\admin\Desktop\work\BarriGuard\tmp\1749_202609040100.mp4")
OUT = Path(__file__).parent

WIN_S = 180.0
STRIDE = 2
FLASH_TH = 60
ON_PX = 8
ROI_R = 10
SPOT_MIN_A = 6
MERGE_R = 10
DOME_STATIC_FRAC = 0.60
ALIVE_R = 18
ALIVE_DY = 12
LAG_LO, LAG_HI = 5, 12
AC_TH = 0.35
DUTY_LO, DUTY_HI = 0.05, 0.85
WIN_STABLE = 0.60
H, W = 1080, 1920

BANDS = {
    "near": (0, 380, 1100, 720),
    "far": (1100, 360, 1920, 470),
    "right": (1100, 470, 1920, 1065),
}


def in_band(x, y):
    for nm, (x0, y0, x1, y1) in BANDS.items():
        if x0 <= x < x1 and y0 <= y < y1:
            return nm
    return None


def clamp_roi(x, y, r):
    x0 = int(np.clip(x - r, 0, W - 2 * r))
    y0 = int(np.clip(y - r, 0, H - 2 * r))
    return x0, y0


def main():
    t0 = time.time()
    cap = cv.VideoCapture(str(VID))
    fps = cap.get(cv.CAP_PROP_FPS)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    win_frames = (int(WIN_S * fps) // 2) * 2
    win_samples = win_frames // STRIDE

    spots = {}
    dome_acc = np.zeros((H, W), np.uint16)
    amb_acc = np.zeros((H, W), np.uint16)
    amb_vsum = np.zeros((H, W), np.float32)
    stat_n = 0
    ref = None
    fi = 0
    while fi < total:
        win_id = fi // win_frames
        mx = np.zeros((H, W), np.uint8)
        hist = np.zeros((16, H, W), np.uint16)
        patches = {k: [] for k in spots}
        f = fi
        while f < total and f // win_frames == win_id:
            ok = cap.grab()
            f += 1
            if not ok:
                break
            if (f - 1) % STRIDE:
                continue
            ok, fr = cap.retrieve()
            if not ok:
                break
            if ref is None:
                ref = fr.copy()
            g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
            np.maximum(mx, g, out=mx)
            b = g >> 4
            for kb in range(16):
                hist[kb][b == kb] += 1
            hsv = cv.cvtColor(fr, cv.COLOR_BGR2HSV)
            hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            dome_acc += ((hh >= 12) & (hh <= 32) & (ss >= 60) & (ss <= 175) &
                         (vv >= 70) & (vv <= 190))
            amb = ((hh >= 12) & (hh <= 28) & (ss >= 140) & (vv >= 185))
            amb_acc += amb
            amb_vsum += vv * amb
            stat_n += 1
            for k, sp in spots.items():
                pw = sp.setdefault("pat", {}).get(win_id)
                if pw is None:
                    x0, y0 = clamp_roi(sp["x"], sp["y"], ROI_R)
                    pw = {"x0": x0, "y0": y0, "p": []}
                    sp["pat"][win_id] = pw
                pw["p"].append(g[pw["y0"]:pw["y0"] + 2 * ROI_R,
                                 pw["x0"]:pw["x0"] + 2 * ROI_R].copy())
        bg = half_p50_from_hist(hist)
        flash = cv.subtract(mx, bg)
        _, th = cv.threshold(flash, FLASH_TH, 255, cv.THRESH_BINARY)
        th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))
        ncc, lab, stats, cents = cv.connectedComponentsWithStats(th, 8)
        for i in range(1, ncc):
            if stats[i, 4] < SPOT_MIN_A:
                continue
            x, y = float(cents[i][0]), float(cents[i][1])
            hit = None
            bd = MERGE_R * MERGE_R
            for k, sp in spots.items():
                dxx, dyy = sp["x"] - x, sp["y"] - y
                dist = dxx * dxx + dyy * dyy
                if dist < bd:
                    bd = dist
                    hit = k
            if hit:
                spots[hit]["wins"].add(win_id)
                spots[hit]["cent"][win_id] = (x, y)
                spots[hit]["x"] = 0.5 * spots[hit]["x"] + 0.5 * x
                spots[hit]["y"] = 0.5 * spots[hit]["y"] + 0.5 * y
            else:
                spots[(round(x), round(y))] = {"x": x, "y": y, "wins": {win_id},
                                               "cent": {win_id: (x, y)}, "ser": []}
        # series from cached patches (same fixed ROI coords as collection)
        for k, sp in spots.items():
            pw = sp.get("pat", {}).pop(win_id, None)
            if pw and len(pw["p"]) == win_samples:
                arr = np.stack(pw["p"]).astype(np.int16)
                d = arr - bg[pw["y0"]:pw["y0"] + 2 * ROI_R,
                             pw["x0"]:pw["x0"] + 2 * ROI_R][None]
                sp["ser"].append(((d >= FLASH_TH).sum(axis=(1, 2)), win_id))
        print(f"[win {win_id}] spots={len(spots)}  {time.time()-t0:.0f}s", flush=True)
        fi = f

    cap.release()
    import pickle
    spot_dbg = {}
    for k, sp in spots.items():
        if len(sp["ser"]) >= 1:
            spot_dbg[k] = {"x": sp["x"], "y": sp["y"],
                           "ser": [(s.astype(np.uint16), wi) for s, wi in sp["ser"]]}
    with open(OUT / "spots.pkl", "wb") as fh:
        pickle.dump(spot_dbg, fh)
    print(f"saved spots.pkl: {len(spot_dbg)} spots with series")
    dome_frac = dome_acc.astype(np.float32) / max(1, stat_n)
    amb_frac = amb_acc.astype(np.float32) / max(1, stat_n)

    alive_flash = []
    dbg = {"n_ser3": 0, "n_len": 0, "n_duty": 0, "n_wins": 0, "n_acf": 0, "ok": 0}
    for k, sp in spots.items():
        if len(sp["ser"]) < 3:
            continue
        dbg["n_ser3"] += 1
        cnt = np.concatenate([s for s, _ in sp["ser"]])
        if len(cnt) < 0.5 * total / STRIDE:
            continue
        dbg["n_len"] += 1
        on = cnt >= ON_PX
        duty = on.mean()
        if not (DUTY_LO <= duty <= DUTY_HI):
            continue
        dbg["n_duty"] += 1
        w_duty = [(wi, (s >= ON_PX).mean()) for s, wi in sp["ser"]]
        on_wins = sum(1 for _, d in w_duty if d >= 0.03)
        if on_wins < WIN_STABLE * len(w_duty):
            continue
        dbg["n_wins"] += 1
        x = on.astype(float) - on.mean()
        acf = np.correlate(x, x, "full")[len(x) - 1:]
        if acf[0] <= 0:
            continue
        acf /= acf[0]
        seg = acf[LAG_LO:LAG_HI + 1]
        if seg.max() < AC_TH:
            continue
        dbg["n_acf"] += 1
        lag = int(np.argmax(seg)) + LAG_LO
        alive_flash.append({"x": round(sp["x"], 1), "y": round(sp["y"], 1),
                            "duty": round(float(duty), 3), "lag": lag,
                            "ac": round(float(seg.max()), 2), "wins": on_wins})
        dbg["ok"] += 1
    print("alive debug:", dbg)
    # diagnostic for known lamps
    for k, sp in spot_dbg.items():
        if abs(sp["x"] - 1326) < 8 and abs(sp["y"] - 411) < 8:
            cnt = np.concatenate([s for s, _ in sp["ser"]])
            on = cnt >= ON_PX
            print(f"DBG ref1326: ser_n={len(sp['ser'])} cnt_len={len(cnt)} duty={on.mean():.3f} "
                  f"max_on={cnt.max()} mean_on={cnt.mean():.1f}")
    alive_flash.sort(key=lambda r: (in_band(r["x"], r["y"]) or "?", r["x"]))

    steady = []
    n, lab, stats, cents = cv.connectedComponentsWithStats(
        (amb_frac >= 0.80).astype(np.uint8) * 255, 8)
    for i in range(1, n):
        a = int(stats[i, 4])
        if a < 8:
            continue
        x, y = float(cents[i][0]), float(cents[i][1])
        v = amb_vsum[int(y), int(x)] / max(1, amb_acc[int(y), int(x)])
        steady.append({"x": round(x, 1), "y": round(y, 1), "area": a, "v": round(float(v))})

    dome_mask = (dome_frac >= DOME_STATIC_FRAC).astype(np.uint8) * 255
    dome_mask = cv.morphologyEx(dome_mask, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cents = cv.connectedComponentsWithStats(dome_mask, 8)
    dead = []
    for i in range(1, n):
        a = int(stats[i, 4])
        x, y = float(cents[i][0]), float(cents[i][1])
        if a < 3 or a > 60 or y >= 1065:
            continue
        bw, bh = int(stats[i, 2]), int(stats[i, 3])
        fill = a / max(1, bw * bh)
        asp = bw / max(1, bh)
        if fill < 0.45 or not (0.4 <= asp <= 2.5) or in_band(x, y) is None:
            continue
        hit = False
        for r in alive_flash:
            if abs(r["x"] - x) < ALIVE_R and abs(r["y"] - y) < ALIVE_DY:
                hit = True
                break
        if not hit:
            for r in steady:
                dx, dy = abs(r["x"] - x), r["y"] - y
                if (dx < ALIVE_R and abs(dy) < ALIVE_DY) or (dx <= 25 and -55 <= dy <= -5):
                    hit = True
                    break
        if hit:
            continue
        dead.append({"x": round(x, 1), "y": round(y, 1), "area": a,
                     "frac": round(float(dome_frac[int(y), int(x)]), 2),
                     "fill": round(fill, 2), "band": in_band(x, y)})
    dead.sort(key=lambda r: (r["band"], r["x"]))

    print(f"\n== ALIVE flash: {len(alive_flash)}")
    for r in alive_flash:
        print(f"  ({r['x']:7.1f},{r['y']:7.1f}) duty={r['duty']*100:5.1f}% lag={r['lag']} ac={r['ac']:.2f} wins={r['wins']}")
    print(f"== ALIVE steady: {len(steady)}")
    for r in steady:
        print(f"  ({r['x']:7.1f},{r['y']:7.1f}) area={r['area']} V={r['v']}")
    print(f"== DEAD: {len(dead)}")
    for r in dead:
        print(f"  ({r['x']:7.1f},{r['y']:7.1f}) band={r['band']} area={r['area']} frac={r['frac']} fill={r['fill']}")

    vis = ref.copy()
    for r in alive_flash:
        cv.circle(vis, (int(r["x"]), int(r["y"])), 10, (0, 255, 0), 2)
    for r in steady:
        cv.circle(vis, (int(r["x"]), int(r["y"])), 10, (255, 255, 0), 2)
    for r in dead:
        cv.rectangle(vis, (int(r["x"]) - 12, int(r["y"]) - 12),
                     (int(r["x"]) + 12, int(r["y"]) + 12), (0, 0, 255), 2)
        cv.putText(vis, "DEAD", (int(r["x"]) - 20, int(r["y"]) + 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv.imwrite(str(OUT / "pipeline_map.png"), vis)
    json.dump({"alive_flash": alive_flash, "alive_steady": steady, "dead": dead},
              open(OUT / "pipeline_result.json", "w"), ensure_ascii=False)
    print(f"done {time.time()-t0:.0f}s")


def half_p50_from_hist(hist):
    cdf = hist.cumsum(axis=0, dtype=np.uint32)
    tot = np.maximum(hist.sum(axis=0, dtype=np.uint32), 1)
    half = tot // 2 + 1
    med = np.zeros(hist.shape[1:], np.uint8)
    for k in range(16):
        med[(cdf[k] >= half) & (med == 0)] = k * 16 + 8
    return med


if __name__ == "__main__":
    main()
