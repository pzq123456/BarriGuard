"""E6: multi-scale (coarse E5-frozen + fine) + local scale estimate.
Frozen: FLOOR_MAX, d thresholds, ROI, weights, health, hysteresis.
No Hough/pitch/SAM/anchor. Scale only prior, never predicts gap pos.
"""
import cv2
import numpy as np
import json
import csv
from pathlib import Path

from exp_e5 import (masks, col_range, red_m, white_m, band_cols, to812,
                    valid812, smooth as smooth6, LEFT_ROW_NORM, FLOOR_MAX,
                    MIN_WIDTH, WIN, GAP, CX_TOL, REF)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

# --- fine path consts (engineering, not fitted to GT) ---
SIGMA_F = 2.5
BASE_F = 101
MINW_F = 18
COREW_F = 12
CORE_EXIT = 0.06
CORE_MIN_W = 30  # coarse core, E5 value

WHITE1 = (54, 122)
WHITE2 = (196, 248)
BG = (700, 800)
FAR = (450, 700)

def smooth_s(s, v, sigma):
    ss = s.copy()
    if (~v).any() and v.any():
        idx = np.arange(len(s))
        ss[~v] = np.interp(idx[~v], idx[v], s[v])
    k = int(sigma * 6 + 1) | 1
    return cv2.GaussianBlur(ss.reshape(1, -1), (k, 1), sigma).ravel()

def mbase_w(sm, v, win):
    num = cv2.blur((sm * v).reshape(1, -1), (win, 1)).ravel()
    den = cv2.blur(v.astype(np.float32).reshape(1, -1), (win, 1)).ravel()
    mean = num / np.maximum(den, 1e-6)
    tmp = sm.copy().astype(np.float32)
    tmp[~v] = -np.inf
    with np.errstate(invalid="ignore"):
        mx = cv2.dilate(tmp.reshape(1, -1), np.ones((1, win))).ravel()
    mx[~v] = np.nan
    return np.maximum(np.nan_to_num(mean), np.nan_to_num(mx))

def detect_p(sm, bs, v, min_w, core_w):
    d = bs - sm
    outs, i, n = [], 0, len(sm)
    while i < n:
        if (not v[i]) or d[i] <= 0.25:
            i += 1
            continue
        j = i
        while j < n and v[j] and d[j] > 0.125:
            j += 1
        if v[i:j].sum() / max(1, j - i) < 0.9 or (j - i) < min_w:
            i = j
            continue
        env = (int(i), int(j - 1))
        k, best = i, None
        while k < j:
            if sm[k] >= FLOOR_MAX:
                k += 1
                continue
            l = k
            while l < j and sm[l] < FLOOR_MAX + CORE_EXIT:
                l += 1
            e = l
            while e - 1 >= k and sm[e - 1] >= FLOOR_MAX:
                e -= 1
            if e - k >= core_w:
                best = (int(k), int(e - 1), int(e - k), round(float(sm[k:e].min()), 3))
                break
            k = l
        outs.append({"env": env, "core": best, "prom": round(float(d[i:j].max()), 3),
                     "reason": "FLOOR+WIDTH" if best else "FLOOR_FAIL"})
        i = j
    return outs

def scale_h(y0, y1, w0):
    # col height mapped to 812
    h0 = np.where(y0 >= 0, y1 - y0 + 1, 0).astype(np.float32)
    h = cv2.resize(h0.reshape(1, -1), (REF, 1), interpolation=cv2.INTER_LINEAR).ravel()
    return cv2.GaussianBlur(h.reshape(1, -1), (61, 1), 10).ravel()

def scale_redrun(rc):
    # local median red-run width, window +-100; -1 if insufficient
    b = (rc > 0.1).astype(np.uint8)
    runs = []
    i, n = 0, len(b)
    while i < n:
        if not b[i]:
            i += 1
            continue
        j = i
        while j < n and b[j]:
            j += 1
        runs.append((i, j - i))
        i = j
    out = np.full(n, -1.0, np.float32)
    for x in range(n):
        ws = [w for s, w in runs if abs((s + s + w) / 2 - x) < 100]
        if len(ws) >= 2:
            out[x] = float(np.median(ws))
    return out

def zone_rep(sm, bs, v, zone, min_w, core_w):
    c = [o for o in detect_p(sm, bs, v, min_w, core_w)
         if o["env"][0] <= zone[1] and o["env"][1] >= zone[0]]
    if not c:
        seg = sm[zone[0]:zone[1] + 1]
        return {"floor": round(float(seg.min()), 3), "w": 0, "reason": "NO_ENV"}
    o = max(c, key=lambda t: t["env"][1] - t["env"][0])
    w = o["core"][2] if o["core"] else 0
    fl = o["core"][3] if o["core"] else round(float(sm[o["env"][0]:o["env"][1] + 1].min()), 3)
    return {"floor": fl, "w": w, "reason": o["reason"]}

def run_frame(bgr):
    h, w = bgr.shape[:2]
    m = masks(h, w)
    rm = red_m(bgr)
    cov = float(rm[m].mean())
    health = "OK" if cov >= 0.05 else "CAM_BAD"
    y0, y1 = col_range(m)
    rc, wc, vd = band_cols(rm, white_m(bgr), y0, y1)
    v = valid812(vd)
    comb = 0.6 * to812(rc) + 0.4 * to812(wc)
    sc = smooth6(comb, v)
    sf = smooth_s(comb, v, SIGMA_F)
    bc, bf = mbase_w(sc, v, 251), mbase_w(sf, v, BASE_F)
    cc = detect_p(sc, bc, v, MIN_WIDTH, CORE_MIN_W)
    fc = detect_p(sf, bf, v, MINW_F, COREW_F)
    return {"comb": comb, "v": v, "sc": sc, "sf": sf, "bc": bc, "bf": bf,
            "cc": cc, "fc": fc, "health": health, "rc": to812(rc),
            "h": scale_h(y0, y1, w), "rr": scale_redrun(to812(rc))}

def gap_hit(cands):
    return next((c for c in cands if c["core"] and c["core"][0] <= GAP[1] and c["core"][1] >= GAP[0] and c["reason"] == "FLOOR+WIDTH"), None)

if __name__ == "__main__":
    base = ROOT.parent
    refs = [base / "data/input/1749_1333.png", base / "data/input/1749_0944.png"]
    sub = sorted((base / "tmp/cap_day").glob("cap_*.png"))[::10][:12]
    det = []
    for p in refs + sub:
        r = run_frame(cv2.imread(str(p)))
        row = {"img": p.name, "health": r["health"],
               "nc_c": len(r["cc"]), "nc_f": len(r["fc"]),
               "hit_c": gap_hit(r["cc"]) is not None, "hit_f": gap_hit(r["fc"]) is not None}
        for zn, z in [("GAP", GAP), ("W1", WHITE1), ("W2", WHITE2), ("BG", BG), ("FAR", FAR)]:
            a = zone_rep(r["sc"], r["bc"], r["v"], z, MIN_WIDTH, CORE_MIN_W)
            b = zone_rep(r["sf"], r["bf"], r["v"], z, MINW_F, COREW_F)
            row[f"{zn}_c"] = f"{a['floor']}/{a['w']}/{a['reason']}"
            row[f"{zn}_f"] = f"{b['floor']}/{b['w']}/{b['reason']}"
        det.append(row)
    with open(OUT / "E6_det.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(det[0].keys()))
        w.writeheader()
        w.writerows(det)
    # scale table from ref frame
    r0 = run_frame(cv2.imread(str(refs[0])))
    with open(OUT / "E6_scale.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "scale_h", "scale_redrun"])
        for x in range(0, REF, 4):
            w.writerow([x, round(float(r0["h"][x]), 1), round(float(r0["rr"][x]), 1)])
    # coarse/fine/final curves plot
    plot = np.zeros((240, REF, 3), np.uint8)
    for arr, col in [(r0["sc"], (0, 255, 0)), (r0["bc"], (255, 255, 0)),
                     (np.clip(r0["sf"], 0, 1), (255, 0, 255)), (np.clip(r0["bf"], 0, 1), (255, 128, 0))]:
        yv = (np.clip(arr, 0, 1) * 200 + 10).astype(int)
        for i in range(1, REF):
            cv2.line(plot, (i - 1, 240 - yv[i - 1]), (i, 240 - yv[i]), col, 1)
    g = gap_hit(r0["cc"]) or gap_hit(r0["fc"])
    if g:
        cv2.rectangle(plot, (g["core"][0], 0), (g["core"][1], 239), (0, 0, 255), 2)
    cv2.imwrite(str(OUT / "E6_curves.png"), plot)
    print(json.dumps({
        "n": len(det), "hit_c": sum(1 for r in det if r["hit_c"]), "hit_f": sum(1 for r in det if r["hit_f"]),
        "hit_either": sum(1 for r in det if r["hit_c"] or r["hit_f"]),
        "w1_fine_fp": sum(1 for r in det if "FLOOR+WIDTH" in r["W1_f"]),
        "w2_fine_fp": sum(1 for r in det if "FLOOR+WIDTH" in r["W2_f"]),
    }, indent=1))
    print(json.dumps(det[0], indent=1))
