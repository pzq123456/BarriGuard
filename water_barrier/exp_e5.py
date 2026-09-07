"""E5: envelope/core split + full cap_day continuous run.
Frozen: ROI, red-bottom/white-top, FLOOR_MAX, MIN_WIDTH, no anchor/pitch/Hough.
"""
import cv2
import numpy as np
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

LEFT_ROW_NORM = [(0.0, 0.42), (0.72, 0.30), (0.72, 0.60), (0.0, 0.82)]
FLOOR_MAX = 0.18  # E4 engineering margin; not fitted
MIN_WIDTH = 40
CORE_MIN_WIDTH = 30  # core is narrower than envelope by nature; WHITE has no sub-floor dip at any width
SMOOTH_SIGMA = 6
BASE_WIN = 251
HEALTH_MIN = 0.05
CX_TOL = 30
REF = 812
WIN = 12  # temporal window on S(x)
GAP = (334, 365)

def masks(h, w):
    pts = np.array([[x * w, y * h] for x, y in LEFT_ROW_NORM], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m > 0

def col_range(mask, minh=20):
    h, w = mask.shape
    y0 = np.full(w, -1)
    y1 = np.full(w, -1)
    for x in range(w):
        ys = np.where(mask[:, x])[0]
        if len(ys) >= minh:
            y0[x], y1[x] = int(ys[0]), int(ys[-1])
    return y0, y1

def red_m(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    r = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255])) | cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    return r > 0

def white_m(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 1] <= 70) & (hsv[:, :, 2] >= 150)

def band_cols(red, white, y0, y1):
    w = len(y0)
    rc = np.zeros(w, np.float32)
    wc = np.zeros(w, np.float32)
    vd = np.zeros(w, bool)
    for x in range(w):
        if y0[x] < 0:
            continue
        vd[x] = True
        hh = y1[x] - y0[x] + 1
        cut = y0[x] + int(hh * 0.6)
        rc[x] = float(red[cut:y1[x] + 1, x].mean())
        wc[x] = float(white[y0[x]:cut, x].mean())
    return rc, wc, vd

def to812(s):
    return cv2.resize(s.reshape(1, -1), (REF, 1), interpolation=cv2.INTER_LINEAR).ravel()

def valid812(vd):
    v = cv2.resize(vd.astype(np.uint8).reshape(1, -1), (REF, 1), interpolation=cv2.INTER_NEAREST).ravel() > 0
    return cv2.erode(v.astype(np.uint8).reshape(1, -1), np.ones((1, 7))).ravel() > 0

def mblur(s, v):
    num = cv2.blur((s * v).reshape(1, -1), (BASE_WIN, 1)).ravel()
    den = cv2.blur(v.astype(np.float32).reshape(1, -1), (BASE_WIN, 1)).ravel()
    return num / np.maximum(den, 1e-6)

def mbase(sm, v):
    mean = mblur(sm, v)
    tmp = sm.copy().astype(np.float32)
    tmp[~v] = -np.inf
    with np.errstate(invalid="ignore"):
        mx = cv2.dilate(tmp.reshape(1, -1), np.ones((1, BASE_WIN))).ravel()
    mx[~v] = np.nan
    return np.maximum(np.nan_to_num(mean), np.nan_to_num(mx))

def smooth(s, v):
    ss = s.copy()
    if (~v).any() and v.any():
        idx = np.arange(len(s))
        ss[~v] = np.interp(idx[~v], idx[v], s[v])
    k = SMOOTH_SIGMA * 6 + 1
    return cv2.GaussianBlur(ss.reshape(1, -1), (k, 1), SMOOTH_SIGMA).ravel()

def detect(sm, bs, v):
    # L1 envelope: discovery d>0.25, extend d>0.125; L2 core: S<FLOOR_MAX
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
        # core inside envelope, hysteresis: enter FLOOR_MAX, exit +0.06
        k = i
        best = None
        while k < j:
            if sm[k] >= FLOOR_MAX:
                k += 1
                continue
            l = k
            while l < j and sm[l] < FLOOR_MAX + 0.06:
                l += 1
            # trim trailing above-floor padding
            e = l
            while e - 1 >= k and sm[e - 1] >= FLOOR_MAX:
                e -= 1
            if e - k >= CORE_MIN_WIDTH:
                best = (int(k), int(e - 1), int(e - k), round(float(sm[k:e].min()), 3))
                break
            k = l
        # envelope still gated by MIN_WIDTH above; core by CORE_MIN_WIDTH
        reason = "FLOOR+WIDTH" if best else "FLOOR_FAIL"
        outs.append({"env": env, "core": best, "prom": round(float(d[i:j].max()), 3), "reason": reason})
        i = j
    return outs

if __name__ == "__main__":
    base = ROOT.parent
    seq = sorted((base / "tmp/cap_day").glob("cap_*.png"))  # continuous run: cap_day only
    refs = [base / "data/input/1749_1333.png", base / "data/input/1749_0944.png"]
    # reference frames: single-frame check (no temporal), E4 acceptance A
    for p in refs:
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        m = masks(h, w)
        rm = red_m(bgr)
        y0, y1 = col_range(m)
        rc, wc, vd = band_cols(rm, white_m(bgr), y0, y1)
        v = valid812(vd)
        comb = 0.6 * to812(rc) + 0.4 * to812(wc)
        sm0 = smooth(comb, v)
        c0 = detect(sm0, mbase(sm0, v), v)
        gap0 = next((c for c in c0 if c["core"] and c["core"][0] <= GAP[1] and c["core"][1] >= GAP[0] and c["reason"] == "FLOOR+WIDTH"), None)
        print("REF", p.name, gap0["core"] if gap0 else "MISS", [ (c["env"], c["core"], c["reason"]) for c in c0 ])
    S_hist, V_hist, per = [], [], []
    state, suspect, miss, prev_cx = "NORMAL", 0, 0, None
    for p in seq:
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        m = masks(h, w)
        rm = red_m(bgr)
        cov = float(rm[m].mean())
        n_roi = int(m.sum())
        health = "OK" if cov >= HEALTH_MIN else "CAM_BAD"
        scene = "OK"
        if n_roi == 0 or (m.sum() / m.size) < 0.02:
            scene = "SCENE_INVALID"
        y0, y1 = col_range(m)
        rc, wc, vd = band_cols(rm, white_m(bgr), y0, y1)
        v = valid812(vd)
        comb = 0.6 * to812(rc) + 0.4 * to812(wc)
        S_hist.append(comb)
        V_hist.append(v)
        # causal temporal median over last WIN
        Ws = np.stack(S_hist[-WIN:])
        med = np.median(Ws, axis=0)
        vmaj = (np.stack(V_hist[-WIN:]).mean(axis=0) >= 0.9)  # fixed semantics
        smm = smooth(med, vmaj)
        cands = detect(smm, mbase(smm, vmaj), vmaj) if health == "OK" and scene == "OK" else []
        # gap = core overlapping GT (eval only)
        gap = next((c for c in cands if c["core"] and c["core"][0] <= GAP[1] and c["core"][1] >= GAP[0] and c["reason"] == "FLOOR+WIDTH"), None)
        cx = (gap["core"][0] + gap["core"][1]) / 2 if gap else None
        if cx is not None and prev_cx is not None and abs(cx - prev_cx) < CX_TOL:
            suspect += 1
            miss = 0
        elif cx is not None:
            suspect = 1
            miss = 0
        else:
            miss += 1
            if miss >= 2:
                suspect = 0
        prev_cx = cx if cx is not None else prev_cx
        st = "OPEN" if suspect >= 2 else ("SUSPECT" if suspect >= 1 else "NORMAL")
        if health != "OK":
            st = "CAM_BAD"
        elif scene != "OK":
            st = "SCENE_INVALID"
        elif st == "NORMAL" and suspect >= 2 - 1 and miss < 2 and prev_cx is not None and len(per) and per[-1]["state"] == "OPEN":
            st = "OPEN"  # single-miss hysteresis
            suspect = max(suspect, 2)
        state = st
        per.append({"img": p.name, "env": gap["env"] if gap else "", "core": gap["core"] if gap else "",
                    "prom": gap["prom"] if gap else "", "reason": gap["reason"] if gap else "NO_CAND",
                    "health": health, "scene": scene, "state": state,
                    "ncand": len(cands)})
    with open(OUT / "E5_run.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["img", "env", "core", "prom", "reason", "health", "scene", "state", "ncand"])
        w.writeheader()
        w.writerows(per)
    # stats: persistence, latency, jitter (core only)
    cores = [(c["core"][0], c["core"][1]) for c in [next((x for x in per if x["img"] == p.name), None) for p in seq] if c and c["core"] != ""]
    # recompute from per directly
    cc = [r for r in per if isinstance(r["core"], tuple)]
    cxv = [(r["core"][0] + r["core"][1]) / 2 for r in cc]
    wdv = [r["core"][2] for r in cc]
    opens = sum(1 for r in per if r["state"] == "OPEN")
    first_open = next((i for i, r in enumerate(per) if r["state"] == "OPEN"), -1)
    trans = sum(1 for i in range(1, len(per)) if per[i]["state"] != per[i - 1]["state"])
    print(json.dumps({
        "n": len(per), "open_frac": round(opens / len(per), 3), "latency_frames": first_open,
        "transitions": trans, "miss_frames": sum(1 for r in per if r["reason"] == "NO_CAND"),
        "cx_mean": round(float(np.mean(cxv)), 1) if cxv else -1, "cx_std": round(float(np.std(cxv)), 2) if cxv else -1,
        "w_mean": round(float(np.mean(wdv)), 1) if wdv else -1, "w_std": round(float(np.std(wdv)), 2) if wdv else -1,
        "gap_hit_frames": len(cc)}, indent=1))
    print(json.dumps({"tail": per[-3:]}, indent=1, ensure_ascii=False))
