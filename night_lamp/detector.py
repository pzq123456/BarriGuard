"""Frozen overnight detector (frozen-20260908) -- EXTRACTED, NOT IMPROVED.

Verbatim math from overnight_run.py (20260908-snap1):
  ON rule : 7x7max >= ROI median+40 OR 7x7mean >= ROI median+30
  A: n_on >= 1
  B: AC(lag5..12) >= 0.35 AND 0.05 <= duty <= 0.85
  C: B AND n_on >= 2 AND AC >= 0.50 (observational subset only)

Only change vs overnight_run: thresholds come from config dict instead of
module constants. Defaults below equal the frozen constants, so default
behavior is bit-identical. No DEAD/SUSPECT machine here -- profiles are
recorded, alerting policy lives outside this module.
"""
import cv2 as cv
import numpy as np

DETECTOR_VERSION = ("frozen-20260908: ON=7x7max>=med+40|7x7mean>=med+30; "
                    "A=n_on>=1; B=AC(lag5-12)>=0.35&duty[0.05,0.85]; "
                    "C=B&n_on>=2&AC>=0.50(obs)")

K7 = np.ones((7, 7), np.uint8)

# Frozen defaults (must match configs/config_1749.yaml detector block).
ROI_R = 10
ON_MAX_DELTA = 40
ON_MEAN_DELTA = 30
LAG_LO, LAG_HI = 5, 12
AC_TH, DUTY_LO, DUTY_HI = 0.35, 0.05, 0.85
C_AC, C_MIN_ON = 0.50, 2


def ac_score(on, lag_lo=LAG_LO, lag_hi=LAG_HI):
    x = on.astype(float) - on.mean()
    if float(x @ x) <= 0:
        return 0.0, -1
    acf = np.correlate(x, x, "full")[len(x) - 1:]
    if acf[0] <= 0:
        return 0.0, -1
    acf = acf / acf[0]
    seg = acf[lag_lo:lag_hi + 1]
    return round(float(seg.max()), 3), int(np.argmax(seg)) + lag_lo


def lamp_metrics(frames, x, y, roi_r=ROI_R,
                 on_max_delta=ON_MAX_DELTA, on_mean_delta=ON_MEAN_DELTA,
                 lag_lo=LAG_LO, lag_hi=LAG_HI,
                 ac_th=AC_TH, duty_lo=DUTY_LO, duty_hi=DUTY_HI,
                 c_ac=C_AC, c_min_on=C_MIN_ON):
    """Per-burst evidence for one lamp. frames: list of gray images."""
    H, W = frames[0].shape
    x0 = int(np.clip(round(x) - roi_r, 0, W - 2 * roi_r))
    y0 = int(np.clip(round(y) - roi_r, 0, H - 2 * roi_r))
    sig, on, meds = [], [], []
    for g in frames:
        roi = g[y0:y0 + 2 * roi_r, x0:x0 + 2 * roi_r]
        med = float(np.median(roi))
        mx7 = float(cv.dilate(roi, K7).max())
        mn7 = float(cv.blur(roi, (7, 7)).max())
        sig.append(mx7 - med)
        meds.append(med)
        on.append(bool(mx7 >= med + on_max_delta or mn7 >= med + on_mean_delta))
    sig = np.array(sig)
    on = np.array(on)
    n_on = int(on.sum())
    duty = round(float(on.mean()), 4)
    ac, lag = ac_score(on, lag_lo, lag_hi)
    peak = round(float(sig.max()), 1)
    pf = int(np.argmax(sig))
    pw = int((sig >= 0.5 * peak).sum()) if peak > 0 else 0
    return {"n_valid": len(frames), "n_on": n_on, "duty": duty, "ac": ac, "lag": lag,
            "min": round(float(sig.min()), 1), "median": round(float(np.median(sig)), 1),
            "max": round(float(sig.max()), 1), "p90": round(float(np.percentile(sig, 90)), 1),
            "peak": peak, "peak_frame": pf, "peak_width": pw,
            "roi_median": round(float(np.median(meds)), 1),
            "profile_A": int(n_on >= 1),
            "profile_B": int(ac >= ac_th and duty_lo <= duty <= duty_hi),
            "profile_C": int(ac >= ac_th and duty_lo <= duty <= duty_hi
                              and n_on >= c_min_on and ac >= c_ac)}
