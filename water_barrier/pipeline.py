"""water_gap 算法主体 (实现 server.Algorithm 协议): 帧 -> 缺口事件 + 标注 + 调试图层。

分层: BarrierAlarm 是逐帧候选 + support 累计 (标定由构造参数注入)；
本模块做轨道匹配 + track.py 的"衰减池 + 路露率 RER 确认 + 迟滞状态机"，
把"真移除"与"人/车长期遮挡"分开 (单帧无法分离，靠时间+RER)。

RER 的"水马前景"改用红/白颜色掩膜 (轴向采样不输出分割图)，
语义不变: 缺口框内非水马像素中真实路面 patch 占比。

对外只出三样东西 (见 server/algo.py):
  events  非 NORMAL 轨道 (alarm/suspected, 含 box/severity/rer/row_id)
  annots  全部轨道的框标注 (含 NORMAL, 供叠加渲染)
  debug   {"roi_mask": ROI 并集掩膜} (按需加层, server 原样 serving)
"""
from dataclasses import dataclass
from typing import List

import cv2 as cv
import numpy as np

from server.algo import AlgoResult, Annotation, Event

from .alarm import BarrierAlarm
from .detector import red_m, white_m
from .track import GapCondition, GapState, GapTracker, RoadColorProfile, calc_patch_rer

DETECT_KEYS = ("trim", "conf_th", "flank_win", "flank_th",
               "support_th", "min_box_width")
ROW_DETECT_KEYS = ("floor_max", "core_exit", "min_width", "core_min_width")

LEVEL = {GapState.ALARM: "alarm", GapState.SUSPECTED: "suspected",
         GapState.NORMAL: "info"}


@dataclass
class TrackView:
    box: tuple
    state: GapState
    severity: float
    kind: str
    rer: float
    row_id: str = ""


def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    aa = (ax1 - ax0) * (ay1 - ay0)
    ab = (bx1 - bx0) * (by1 - by0)
    return inter / float(aa + ab - inter)


class WaterGapAlgorithm:
    """water_gap per_frame 算法, 状态为实例属性 (多流各持一份)。"""

    name = "water_gap"
    cadence = "per_frame"

    def __init__(self, frame_shape: tuple, calib: dict, camera_id: str = "") -> None:
        for k in calib.get("detect", {}):
            if k not in DETECT_KEYS:
                raise ValueError(f"未知detect键: {k}")
        self._cam = camera_id
        self._tcfg = calib["track"]
        self._road_rects = calib["road_rois"]
        self._detect = calib.get("detect", {})
        rows = calib["rows"]
        row_detect = {}
        for r in rows:
            for k in r.get("detect", {}):
                if k not in ROW_DETECT_KEYS:
                    raise ValueError(f"未知行detect键 {r['id']}.{k}")
            if r.get("detect"):
                row_detect[r["id"]] = dict(r["detect"])
        polys = [np.asarray(r["poly"], dtype=np.float32) for r in rows]
        self._alarm = BarrierAlarm(
            polys=polys, row_ids=[r["id"] for r in rows],
            units={r["id"]: r["U"] for r in rows},
            report_rows={r["id"] for r in rows if r.get("report")},
            row_detect=row_detect, **self._detect)
        self._tracks: List[GapTracker] = []
        self.fg = _roi_mask(frame_shape, self._alarm.polys)

    def reset(self) -> None:
        self._tracks = []
        self._alarm.reset()

    def _confirm(self, track: GapTracker):
        """构建该缺口的 RER 确认闭包：中值帧上路色建模 + 框内路露率。"""
        last_good = {"rer": 0.0}

        def confirm(median_bgr):
            try:
                x0, y0, x1, y1 = track.box
                sm = np.zeros(median_bgr.shape[:2], np.uint8)
                sm[max(0, y0):y1, max(0, x0):x1] = 255
                crops = [median_bgr[ry0:ry1, rx0:rx1] for rx0, ry0, rx1, ry1 in self._road_rects]
                crops = [c for c in crops if c.size > 0]
                if not crops:
                    return last_good["rer"]
                min_w = min(c.shape[1] for c in crops)
                if min_w <= 0:
                    return last_good["rer"]
                profile = RoadColorProfile(np.vstack([c[:, :min_w] for c in crops]))
                fg = np.where(red_m(median_bgr) | white_m(median_bgr),
                              255, 0).astype(np.uint8)
                rer = calc_patch_rer(median_bgr, sm, fg, profile,
                                     self._tcfg["rer_purity"], self._tcfg["patch_size"])
                last_good["rer"] = rer
                return rer
            except Exception:
                return last_good["rer"]

        return confirm

    def step(self, frame: np.ndarray, now: float) -> AlgoResult:
        """处理一帧：报警算法 -> 匹配轨道 -> 更新状态机。

        SUNGLARE 只定状态（经 debug.frame_status 透出，UNKNOWN，不是 NORMAL），
        不冻结检测：眩光帧的真缺口仍可能经 RER 确认，误报仍由 RER 挡。
        """
        tcfg = self._tcfg
        road_white = _road_white_frac(frame, self._road_rects)
        status = "SUNGLARE" if (self._road_rects
                                and road_white > tcfg["sunglare_road_white"]) else "OK"
        gaps = self._alarm.step(frame)

        matched = {t: False for t in self._tracks}
        for g in gaps:
            hit = False
            for t in self._tracks:
                if _match(g.box, t.box, tcfg):
                    t.box = g.box
                    t.severity, t.kind, t.row_id = g.support, g.kind, g.row_id
                    t.last_seen = now
                    t.push_frame(frame)
                    t.update(GapCondition.DEFECTIVE, now, self._confirm(t))
                    matched[t] = True
                    hit = True
                    break
            if not hit:
                t = GapTracker(g.box, tcfg)
                t.severity, t.kind, t.row_id = g.support, g.kind, g.row_id
                t.last_seen = now
                t.push_frame(frame)
                t.update(GapCondition.DEFECTIVE, now, self._confirm(t))
                self._tracks.append(t)
                matched[t] = True

        for t in self._tracks:
            if not matched[t]:
                t.update(GapCondition.INTACT, now, self._confirm(t))

        self._tracks = [t for t in self._tracks
                        if now - t.last_seen < tcfg["track_stale_s"]
                        or t.state != GapState.NORMAL]
        views = [TrackView(t.box, t.state, t.severity, t.kind, t._rer_cache[1], t.row_id)
                 for t in self._tracks]
        annots = [Annotation("box", v.box, f"{v.kind}:{v.state.name}", LEVEL[v.state],
                             {"severity": v.severity, "rer": round(float(v.rer), 3),
                              "row_id": v.row_id})
                  for v in views]
        events = [Event(self._cam, "water_gap", now,
                        "alarm" if v.state == GapState.ALARM else "suspected",
                        {"box": v.box, "severity": v.severity, "kind": v.kind,
                         "rer": round(float(v.rer), 3), "row_id": v.row_id})
                  for v in views if v.state != GapState.NORMAL]
        return AlgoResult(events, annots, {"roi_mask": self.fg, "frame_status": status,
                                           "road_white": round(road_white, 3)})


def _road_white_frac(frame: np.ndarray, rects) -> float:
    """路面取色框的 white 占比中位数（日光门控用；无框时返回 0 即不过门）。"""
    vals = []
    for rx0, ry0, rx1, ry1 in rects:
        crop = frame[max(0, ry0):ry1, max(0, rx0):rx1]
        if crop.size > 0:
            vals.append(float((white_m(crop)).mean()))
    return float(np.median(vals)) if vals else 0.0


def _roi_mask(frame_shape: tuple, polys) -> np.ndarray:
    """ROI 并集掩膜（debug 图层用；静态，每流计算一次）。"""
    h, w = frame_shape[:2]
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv.fillPoly(m, [(np.asarray(p, dtype=np.float32) * [w, h]).astype(int)], 255)
    return m


def _match(a: tuple, b: tuple, tcfg: dict) -> bool:
    if _iou(a, b) < tcfg["match_iou"]:
        return False
    gcx, gcy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    tcx, tcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    cd = ((gcx - tcx) ** 2 + (gcy - tcy) ** 2) ** 0.5
    return cd < tcfg["match_center_ratio"] * max(a[2] - a[0], b[2] - b[0])
