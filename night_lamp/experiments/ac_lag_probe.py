"""Lag-limited AC probe: equivalence check + 1750 headlight-wash curve.

Two questions, one script, hardcoded on purpose (experiment, no CLI):
  1. does ac_limited reproduce detector.ac_score on the shared lag window?
  2. what does the lag-limited AC curve look like on the 1750 wash (a
     one-off transit) versus registered lamps (periodic flash)?
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from night_lamp import detector  # noqa: E402
from night_lamp.periodicity import ac_limited  # noqa: E402
from night_lamp.tools import nightly_map as nm  # noqa: E402

VID = ROOT / "tmp" / "1002491_night_10min.mp4"  # camera 1750, 12.49fps
COARSE_STEP = 6
FINE_STEP = 1
DELTA = 40
LAG_S = (0.3, 2.6)
K_WASH = 6
LAMPS_1750 = {"M02": (339.4, 176.1), "M06": (783.9, 111.2),
              "M10": (1305.4, 118.9)}


def equivalence():
    rng = np.random.default_rng(0)
    max_dpk, lag_bad = 0.0, 0
    for _ in range(300):
        n = int(rng.integers(60, 500))
        on = (rng.random(n) < rng.uniform(0.05, 0.7)).astype(np.uint8)
        peak, lag, _ = ac_limited(on, detector.LAG_LO, detector.LAG_HI)
        ref_peak, ref_lag = detector.ac_score(on, detector.LAG_LO,
                                              detector.LAG_HI)
        max_dpk = max(max_dpk, abs(round(peak, 3) - ref_peak))
        if (peak, lag) != (0.0, -1) and lag != ref_lag:
            lag_bad += 1
    print("[equivalence] 300 random binary series vs detector.ac_score:")
    print("  max |round(peak,3)-ref_peak| = %.4g   lag mismatches = %d"
          % (max_dpk, lag_bad))


def main():
    equivalence()

    cap = cv.VideoCapture(str(VID))
    fps = cap.get(cv.CAP_PROP_FPS) or 12.49
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap.release()
    mask = nm.osd_mask(h, w)
    lag_lo = max(1, int(round(LAG_S[0] * fps / FINE_STEP)))
    lag_hi = int(round(LAG_S[1] * fps / FINE_STEP))
    print("\n[probe] fps=%.2f total=%d  lag window=[%d,%d] samples"
          % (fps, total, lag_lo, lag_hi))

    meds, _ = nm.med_series(str(VID), COARSE_STEP)
    night_med = float(np.median(meds))
    base = nm.night_base(str(VID), nm.base_index(total, 60), mask, night_med)
    duty, swing, _ = nm.duty_swing(str(VID), base, COARSE_STEP, meds,
                                   night_med, mask, DELTA)
    dyn, lab, stats, _ = nm.split_comps(duty)
    big = [c for c in range(1, len(stats))
           if stats[c, cv.CC_STAT_AREA] >= nm.DYN_MIN_AREA]
    big.sort(key=lambda c: -stats[c, cv.CC_STAT_AREA])
    print("  dyn_frac=%.3f  dyn_comps=%d  largest_area=%d"
          % (dyn.mean(), len(big),
             stats[big[0], cv.CC_STAT_AREA] if big else 0))

    pts, tags = [], []
    if big:
        c = big[0]
        ys, xs = np.nonzero(dyn & (lab == c))
        order = np.argsort(-swing[ys, xs])
        for k in np.linspace(0, len(order) - 1, K_WASH).astype(int):
            pts.append((int(ys[order[k]]), int(xs[order[k]])))
            tags.append("wash")
    for name, (lx, ly) in LAMPS_1750.items():
        pts.append((int(round(ly)), int(round(lx))))
        tags.append(name)

    meds1, _ = nm.med_series(str(VID), FINE_STEP)
    night_med1 = float(np.median(meds1))
    series = nm.series_for(str(VID), FINE_STEP, pts, meds1, night_med1, mask)
    print("\n   x     y    tag   duty  n_on  onsets  lag  peak")
    for p, (y, x) in enumerate(pts):
        on = (series[p].astype(np.float64) - base[y, x]) >= DELTA
        peak, lag, curve = ac_limited(on, lag_lo, lag_hi)
        onsets = int(np.count_nonzero(on[1:] & ~on[:-1])) + int(on[0])
        print(" %4d  %4d  %-5s %.3f %5d %6d  %3d  %.3f"
              % (x, y, tags[p], on.mean(), int(on.sum()), onsets, lag, peak))
        print("    " + " ".join("%.2f" % v for v in curve))


if __name__ == "__main__":
    main()
