"""Additive models -- NOT part of the frozen detector (detector.py untouched).

Provisional 2026-09-11, camera 1749 only, from two night videos
(1749_202609040100.mp4 + SSSS_1749_1_20260909235959_20260910001000.mp4).
Do NOT copy these numbers to a new camera without re-estimation.

F1 adaptive ROI: roi_r = clip(round(min(w,h)/2)-1, 3, 10). Missing w/h
  (watchlist, legacy) falls back to the frozen base (10) bit-identically.
  Basis: 20x20 ROI swallows neighbours on the dense far row (3-15px gaps;
  M13 OFF frame roi_max=254 from a neighbour while its own center is 73),
  while shrinking globally breaks isolated lamps (M06 duty 0.30->0.10 at r=4).

F2 center-swing flash: center 3x3 mean series (neighbour-immune), swing =
  p90-p10. ABSOLUTE gate first: swing < floor -> F2=0 and duty2/ac2 forced 0,
  so the relative on2 rule can never fabricate duty on a dead cell --
  M07-OLD dead (human-confirmed 2026-09-11): swing 8.4 -> gated;
  M07-NEW alive: swing ~120 -> evaluated. on2 = v >= p10+0.4*swing
  (blind_test.py form); duty2/ac2/profile_F2 reuse the frozen AC window
  and duty bounds, no new tuning.

F3 steady record: profile_S = all-ON and swing<25 and roi_median>140.
  Record only (basis: M01/M03/M04 vs M06, same two videos). Never alerts.
"""
import numpy as np

from detector import ac_score

SWING_FLOOR = 20.0
STEADY_SWING_LO = 25.0
STEADY_ROI_HI = 140.0


def adaptive_roi(w, h, base=10):
    """Per-lamp ROI half-size from registry box; base on missing w/h."""
    try:
        m = min(float(w), float(h))
    except (TypeError, ValueError):
        return base
    if not np.isfinite(m) or m <= 0:
        return base
    return int(np.clip(round(m / 2) - 1, 3, 10))


def f2_metrics(frames, x, y, lag_lo=5, lag_hi=12, ac_th=0.35,
               duty_lo=0.05, duty_hi=0.85, swing_floor=SWING_FLOOR):
    """Center 3x3 swing flash evidence. Never raises; returns record dict."""
    xi, yi = int(round(x)), int(round(y))
    v = np.array([float(g[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].mean())
                  for g in frames])
    p10, p90 = float(np.percentile(v, 10)), float(np.percentile(v, 90))
    swing = round(p90 - p10, 1)
    if swing < swing_floor:
        return {"f2_swing": swing, "f2_n_on": 0, "f2_duty": 0.0,
                "f2_ac": 0.0, "f2_lag": -1, "profile_F2": 0,
                "f2_gated": "swing_below_floor"}
    on = v >= p10 + 0.4 * (p90 - p10)
    ac, lag = ac_score(on, lag_lo, lag_hi)
    duty = round(float(on.mean()), 4)
    f2 = int(ac >= ac_th and duty_lo <= duty <= duty_hi)
    return {"f2_swing": swing, "f2_n_on": int(on.sum()), "f2_duty": duty,
            "f2_ac": ac, "f2_lag": lag, "profile_F2": f2,
            "f2_gated": ""}


def steady_flag(n_on, n_valid, swing, roi_median,
                swing_lo=STEADY_SWING_LO, roi_hi=STEADY_ROI_HI):
    """profile_S record-only: bright + flat + always-ON."""
    return int(n_on == n_valid and swing < swing_lo and roi_median > roi_hi)
