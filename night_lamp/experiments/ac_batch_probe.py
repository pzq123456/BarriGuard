"""Batch probe: per-point duty / onsets / AC lag-peak sequence (no verdict).

Feeds the next design step:
  1. does "repeated, equally spaced lag peaks" separate real lamps from
     one-off wash across the whole candidate set (not the 9 hand-picked
     points of ac_lag_probe)?
  2. does the onsets first-stage filter cost us weak/dormant flashers?

Output is a CSV; nothing here classifies. Hardcoded on purpose (experiment).
"""
import csv
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from night_lamp.periodicity import ac_limited  # noqa: E402
from night_lamp.tools import nightly_map as nm  # noqa: E402

VID = ROOT / "tmp" / "1002491_night_10min.mp4"  # camera 1750, 12.49fps
COARSE_STEP = 6
FINE_STEP = 1
DELTA = 40
T_MAX_S = 2.2          # README: flashing period T <= 2.2s
LAG_S = (0.3, 2 * T_MAX_S)  # window >= 2 periods of the slowest lamp
DYN_PER_COMP = 25      # grid samples per dynamic blob
PEAK_FLOOR = 0.2       # local max must clear this to count
OUT_CSV = Path(__file__).parent / "ac_batch.csv"

# registry_1750.csv active fixtures, for eyeballing separability only
LAMPS = {"M01": (152.2, 238.0), "M02": (339.4, 176.1), "M03": (519.4, 154.1),
         "M04": (664.8, 117.7), "M06": (783.9, 111.2), "M07": (1052.2, 115.2),
         "M08": (1204.9, 107.5), "M10": (1305.4, 118.9)}


def local_max_lags(curve, lag_lo, floor):
    """Lags of interior local maxima in `curve` clearing `floor`."""
    out = []
    for i in range(1, len(curve) - 1):
        if curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1] \
                and curve[i] >= floor:
            out.append(lag_lo + i)
    return out


def main():
    cap = cv.VideoCapture(str(VID))
    fps = cap.get(cv.CAP_PROP_FPS) or 12.49
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap.release()
    mask = nm.osd_mask(h, w)
    lag_lo = max(1, int(round(LAG_S[0] * fps / FINE_STEP)))
    lag_hi = int(round(LAG_S[1] * fps / FINE_STEP))

    meds, _ = nm.med_series(str(VID), COARSE_STEP)
    night_med = float(np.median(meds))
    base = nm.night_base(str(VID), nm.base_index(total, 60), mask, night_med)
    duty, swing, _ = nm.duty_swing(str(VID), base, COARSE_STEP, meds,
                                   night_med, mask, DELTA)
    dyn, lab, stats, cand = nm.split_comps(duty)

    pts, tags = [], []
    for c in sorted(cand, key=lambda c: -stats[c, cv.CC_STAT_AREA])[:nm.MAX_CAND]:
        ys, xs = np.nonzero(lab == c)
        j = int(np.argmax(swing[ys, xs]))
        pts.append((int(ys[j]), int(xs[j])))
        tags.append("cand")
    for c in range(1, len(stats)):
        if stats[c, cv.CC_STAT_AREA] < nm.DYN_MIN_AREA:
            continue
        ys, xs = np.nonzero(dyn & (lab == c))
        for k in np.linspace(0, len(ys) - 1,
                             min(DYN_PER_COMP, len(ys))).astype(int):
            pts.append((int(ys[k]), int(xs[k])))
            tags.append("dyn")
    for name, (lx, ly) in LAMPS.items():
        pts.append((int(round(ly)), int(round(lx))))
        tags.append("lamp")

    t0 = time.perf_counter()
    meds1, _ = nm.med_series(str(VID), FINE_STEP)
    night_med1 = float(np.median(meds1))
    t_med = time.perf_counter()
    series = nm.series_for(str(VID), FINE_STEP, pts, meds1, night_med1, mask)
    t_ser = time.perf_counter()

    rows = []
    for p, (y, x) in enumerate(pts):
        on = (series[p].astype(np.float64) - base[y, x]) >= DELTA
        peak, lag, curve = ac_limited(on, lag_lo, lag_hi)
        onsets = int(np.count_nonzero(on[1:] & ~on[:-1])) + int(on[0])
        idx = np.nonzero(on)[0]
        span = 0.0 if idx.size == 0 else (idx[-1] - idx[0]) / on.size
        rows.append({"tag": tags[p], "x": x, "y": y,
                     "duty": round(float(on.mean()), 4),
                     "n_on": int(on.sum()), "onsets": onsets,
                     "span": round(float(span), 3), "peak": round(peak, 3),
                     "lag": lag,
                     "peaks": "|".join(str(v) for v in
                                       local_max_lags(curve, lag_lo,
                                                      PEAK_FLOOR))})
    t_ac = time.perf_counter()

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("wrote %s  (%d pts: %d cand, %d dyn, %d lamp)"
          % (OUT_CSV.name, len(rows), tags.count("cand"), tags.count("dyn"),
             tags.count("lamp")))
    print("lag window=[%d,%d]  peak_floor=%.2f" % (lag_lo, lag_hi, PEAK_FLOOR))
    print("timing: med_series %.1fs | series_for %.1fs (P=%d, N=%d) | "
          "AC loop %.1fs (L=%d, %.2f ms/point)"
          % (t_med - t0, t_ser - t_med, len(pts), len(meds1),
             t_ac - t_ser, lag_hi - lag_lo + 1,
             1000.0 * (t_ac - t_ser) / len(pts)))
    print("\n lamps:")
    for r in rows:
        if r["tag"] == "lamp":
            print("  (%4d,%4d) duty=%.3f onsets=%4d span=%.2f peaks=%s"
                  % (r["x"], r["y"], r["duty"], r["onsets"], r["span"],
                     r["peaks"]))
    for tag in ("cand", "dyn"):
        sel = [r for r in rows if r["tag"] == tag]
        print("\n %s: %d pts | onsets>=2: %d | >=1 local max: %d | "
              ">=2 local max: %d"
              % (tag, len(sel), sum(r["onsets"] >= 2 for r in sel),
                 sum(bool(r["peaks"]) for r in sel),
                 sum(r["peaks"].count("|") >= 1 for r in sel)))


if __name__ == "__main__":
    main()
