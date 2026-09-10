"""Core alarm algorithm: per-frame candidates -> incremental support -> alarm boxes.

Realtime contract for the server framework: construct once, call step() per
frame. No file I/O inside step(). ROI polys + per-row calibration are injected
at construction; defaults = frozen 1749 manual calibration (keep for compat).
New cameras pass explicit row_ids/units/report_rows (see configs/water_gap/).
REF stays global for now (per-row ref_width is a separate calibration task).
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


def flank_score(sm, v, a, b, win=FLANK_WIN, th=FLANK_TH):
    l0, l1 = max(0, a - win), max(0, a - 1)
    r0, r1 = min(REF, b + 1), min(REF, b + win)
    left, right = v[l0:l1], v[r0:r1]
    lm = float(sm[l0:l1][left].mean()) if left.any() else 0.0
    rm = float(sm[r0:r1][right].mean()) if right.any() else 0.0
    return min(1.0, (lm + rm) / (2 * th)), round(lm, 3), round(rm, 3)


def confidence(prom, alen_st, unit, fs):
    """Soft confidence: prom x width gate x flank. Width gate stays hard."""
    if unit > 0 and (alen_st < 0.5 * unit or alen_st > 2.0 * unit):
        return 0.0, "WIDE/NARROW"
    c = min(1.0, max(0.0, prom)) * fs
    return round(c, 3), ("LOW-FLANK" if fs < 1.0 else "OK")


def detect_frame(bgr, polys, row_ids=None, units=None, trim=TRIM,
                 conf_th=CONF_TH, flank_win=FLANK_WIN, flank_th=FLANK_TH,
                 row_detect=None):
    """Stateless single frame -> ({row: [(a, b)]} kept cores, {row: skip reason}).

    row_detect: {行id: {floor_max, core_exit, min_width, core_min_width}} 按行覆盖成核参数。
    """
    rids = list(row_ids) if row_ids is not None else list(ROW_IDS)
    umap = dict(units) if units is not None else U_FROZEN
    if len(polys) != len(rids):
        raise ValueError(f"polys行数({len(polys)})与row_ids({len(rids)})不一致")
    ok, _ = frame_valid(bgr)
    if not ok:
        return {}, {"all": "INVALID"}
    out, skip = {}, {}
    for rid, poly in zip(rids, polys):
        if rid not in umap:
            raise ValueError(f"行缺U标定: {rid}")
        pack = sample_row(bgr, poly)
        valid = apply_end_trim(pack, trim)["v"]
        if is_occluded(pack, valid):
            out[rid] = []
            skip[rid] = "OCCLUDED"
            continue
        unit = umap[rid]
        sm = smooth(pack["s"], valid)
        kept = []
        for c in detect(sm, mbase(sm, valid), valid, **(row_detect or {}).get(rid, {})):
            if not (c["reason"] == "FLOOR+WIDTH" and c["core"]):
                continue
            a, b = c["core"][0], c["core"][1]
            fs, _, _ = flank_score(sm, valid, a, b, flank_win, flank_th)
            conf, _ = confidence(c["prom"], (b - a) / REF * pack["M"], unit, fs)
            if conf >= conf_th:
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

    def __init__(self, polys=None, row_ids=None, units=None, report_rows=None,
                 trim=TRIM, conf_th=CONF_TH, flank_win=FLANK_WIN,
                 flank_th=FLANK_TH, support_th=SUPPORT_TH,
                 min_box_width=MIN_BOX_WIDTH, row_detect=None):
        rids = list(row_ids) if row_ids is not None else list(ROW_IDS)
        if polys is None:
            if rids != list(ROW_IDS):
                raise ValueError("自定义行必须显式传polys")
            polys = load_manual_polys()
        if len(polys) != len(rids):
            raise ValueError(f"polys行数({len(polys)})与row_ids({len(rids)})不一致")
        umap = dict(units) if units is not None else dict(U_FROZEN)
        for rid in rids:
            if rid not in umap:
                raise ValueError(f"行缺U标定: {rid}")
        rep = set(report_rows) if report_rows is not None else set(REPORT_ROWS)
        if not rep.issubset(set(rids)):
            raise ValueError(f"report_rows越界: {sorted(rep)}")
        self._row_ids = rids
        self._polys = list(polys)
        self._units = umap
        self._report = rep
        self._trim = trim
        self._conf_th = conf_th
        self._flank_win = flank_win
        self._flank_th = flank_th
        self._support_th = support_th
        self._min_box_width = min_box_width
        self._row_detect = dict(row_detect) if row_detect else {}
        self.reset()

    def reset(self):
        """清support累计(多流生命周期管理用, 平时不调)。"""
        self._acc = {rid: np.zeros(REF, np.float32) for rid in self._row_ids}
        self._n_obs = {rid: 0 for rid in self._row_ids}
        self._alarms = []

    @property
    def polys(self):
        return list(self._polys)

    @property
    def row_ids(self):
        return list(self._row_ids)

    def step(self, bgr):
        """Feed one frame; return current alarm boxes (list[Gap])."""
        if bgr is None:
            return list(self._alarms)
        h, w = bgr.shape[:2]
        cand, skip = detect_frame(bgr, self._polys, self._row_ids, self._units,
                                  self._trim, self._conf_th,
                                  self._flank_win, self._flank_th,
                                  self._row_detect)
        if "all" in skip:
            return list(self._alarms)
        for rid in self._row_ids:
            if rid in skip:
                continue
            self._n_obs[rid] += 1
            for a, b in cand.get(rid, []):
                self._acc[rid][a:b + 1] += 1
        sup = {rid: self._acc[rid] / max(1, self._n_obs[rid]) for rid in self._row_ids}
        alarms = []
        for rid in self._report:
            pack = sample_row(bgr, self._polys[self._row_ids.index(rid)])
            for a, b in _runs(sup[rid], self._support_th, self._min_box_width):
                alarms.append(Gap(row_id=rid, sta=(a, b),
                                  box=_pixel_box(pack, a, b, w, h),
                                  support=round(float(sup[rid][a:b + 1].max()), 3)))
        self._alarms = alarms
        return list(self._alarms)

    @property
    def support(self):
        return {rid: self._acc[rid] / max(1, self._n_obs[rid]) for rid in self._row_ids}

    @property
    def n_obs(self):
        return dict(self._n_obs)
