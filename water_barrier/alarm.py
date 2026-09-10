"""Core alarm algorithm: per-frame candidates -> incremental support -> alarm boxes.

Realtime contract for the server framework: construct once, call step() per
frame. No file I/O inside step(); ROI polys are injected at construction
(default = frozen manual calibration).
"""

from dataclasses import dataclass

import numpy as np

from .detector import REF, detect, mbase, smooth
from .quality import frame_valid, is_occluded
from .roi import ROW_IDS, load_manual_polys
from .sampling import apply_end_trim, sample_row, sta_to_xy

TRIM = 0.04
CONF_TH = 0.30
FLANK_WIN = 40
FLANK_TH = 0.30

SUPPORT_TH = 0.3
MIN_BOX_WIDTH = 30

# Frozen per-row units (station counts), calibrated once on day frames.
# Recomputing per frame shrinks the gate when red fades, so keep frozen.
U_FROZEN = {"row_0_left_near": 28.5, "row_1_far_center": 3.0, "row_2_right_near": 6.0}
REPORT_ROWS = {"row_0_left_near"}


@dataclass(frozen=True)
class Gap:
    row_id: str
    sta: tuple
    box: tuple
    support: float
    kind: str = "gap"


def flank_score(sm, v, a, b):
    l0, l1 = max(0, a - FLANK_WIN), max(0, a - 1)
    r0, r1 = min(REF, b + 1), min(REF, b + FLANK_WIN)
    left, right = v[l0:l1], v[r0:r1]
    lm = float(sm[l0:l1][left].mean()) if left.any() else 0.0
    rm = float(sm[r0:r1][right].mean()) if right.any() else 0.0
    return min(1.0, (lm + rm) / (2 * FLANK_TH)), round(lm, 3), round(rm, 3)


def confidence(prom, alen_st, unit, fs):
    """Soft confidence: prom x width gate x flank. Width gate stays hard."""
    if unit > 0 and (alen_st < 0.5 * unit or alen_st > 2.0 * unit):
        return 0.0, "WIDE/NARROW"
    c = min(1.0, max(0.0, prom)) * fs
    return round(c, 3), ("LOW-FLANK" if fs < 1.0 else "OK")


def detect_frame(bgr, polys):
    """Stateless single frame -> ({row: [(a, b)]} kept cores, {row: skip reason})."""
    ok, _ = frame_valid(bgr)
    if not ok:
        return {}, {"all": "INVALID"}
    out, skip = {}, {}
    for rid, poly in zip(ROW_IDS, polys):
        pack = sample_row(bgr, poly)
        valid = apply_end_trim(pack, TRIM)["v"]
        if is_occluded(pack, valid):
            out[rid] = []
            skip[rid] = "OCCLUDED"
            continue
        unit = U_FROZEN[rid]
        sm = smooth(pack["s"], valid)
        kept = []
        for c in detect(sm, mbase(sm, valid), valid):
            if not (c["reason"] == "FLOOR+WIDTH" and c["core"]):
                continue
            a, b = c["core"][0], c["core"][1]
            fs, _, _ = flank_score(sm, valid, a, b)
            conf, _ = confidence(c["prom"], (b - a) / REF * pack["M"], unit, fs)
            if conf >= CONF_TH:
                kept.append((a, b))
        out[rid] = kept
    return out, skip


def _pixel_box(pack, a, b, w, h):
    """Station range -> pixel box from measured axis + section y-extent."""
    pts = sta_to_xy(pack, [a, b], w, h)
    x0, x1 = int(round(pts[0][0])), int(round(pts[1][0]))
    top = pack["y0"][a:b + 1]
    bot = pack["y1"][a:b + 1]
    if np.isfinite(top).any() and np.isfinite(bot).any():
        y0, y1 = int(np.nanmin(top)), int(np.nanmax(bot))
    else:
        cy = int(round((pts[0][1] + pts[1][1]) / 2))
        y0, y1 = cy - 20, cy + 20
    x0, x1 = sorted((max(0, x0), min(w - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(h - 1, y1)))
    return (x0, y0, x1, y1)


def _runs(sup, thresh, min_width):
    boxes, i = [], 0
    while i < REF:
        if sup[i] < thresh:
            i += 1
            continue
        j = i
        while j < REF and sup[j] >= thresh:
            j += 1
        if j - i >= min_width:
            boxes.append((int(i), int(j - 1)))
        i = j
    return boxes


class BarrierAlarm:
    """Incremental support accumulator. One instance per stream.

    INVALID/OCCLUDED frames are skipped without counting; the last alarm
    output is frozen (not cleared) while quality gates fail.
    """

    def __init__(self, polys=None):
        self._polys = polys if polys is not None else load_manual_polys()
        self._acc = {rid: np.zeros(REF, np.float32) for rid in ROW_IDS}
        self._n_obs = {rid: 0 for rid in ROW_IDS}
        self._alarms = []

    @property
    def polys(self):
        return list(self._polys)

    def step(self, bgr):
        """Feed one frame; return current alarm boxes (list[Gap])."""
        if bgr is None:
            return list(self._alarms)
        h, w = bgr.shape[:2]
        cand, skip = detect_frame(bgr, self._polys)
        if "all" in skip:
            return list(self._alarms)
        for rid in ROW_IDS:
            if rid in skip:
                continue
            self._n_obs[rid] += 1
            for a, b in cand.get(rid, []):
                self._acc[rid][a:b + 1] += 1
        sup = {rid: self._acc[rid] / max(1, self._n_obs[rid]) for rid in ROW_IDS}
        alarms = []
        for rid in REPORT_ROWS:
            pack = sample_row(bgr, self._polys[ROW_IDS.index(rid)])
            for a, b in _runs(sup[rid], SUPPORT_TH, MIN_BOX_WIDTH):
                alarms.append(Gap(row_id=rid, sta=(a, b),
                                  box=_pixel_box(pack, a, b, w, h),
                                  support=round(float(sup[rid][a:b + 1].max()), 3)))
        self._alarms = alarms
        return list(self._alarms)

    @property
    def support(self):
        return {rid: self._acc[rid] / max(1, self._n_obs[rid]) for rid in ROW_IDS}

    @property
    def n_obs(self):
        return dict(self._n_obs)
