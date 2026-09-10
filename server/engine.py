"""核心引擎：帧 -> 缺口报警算法 -> 时序确认。无固定 slot，无缺口坐标先验。

分层：water_barrier.BarrierAlarm 是核心报警算法（逐帧候选 + support 累计，
阈值冻结在 water_barrier.alarm）；本模块只做框架事：轨道匹配 + 复用
server/track.py 的"衰减池 + 路露率 RER 确认 + 迟滞状态机"做时序收敛，
把"真移除"与"人/车/遮挡"分开（单帧无法分离，靠时间+RER）。

RER 的"水马前景"改用红/白颜色掩膜（water_barrier 轴向采样不输出分割图），
语义不变：缺口框内非水马像素中真实路面 patch 占比。
"""
from dataclasses import dataclass
from typing import List

import cv2 as cv
import numpy as np

from .track import GapCondition, GapState, GapTracker, RoadColorProfile, calc_patch_rer

from water_barrier import BarrierAlarm
from water_barrier.detector import red_m, white_m


@dataclass
class TrackView:
    box: tuple
    state: GapState
    severity: float
    kind: str
    rer: float


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


class Monitor:
    def __init__(self, frame_shape: tuple, algo: dict) -> None:
        self.algo = algo
        self._road_rects = algo["road_rois"]
        polys = ([np.asarray(p, dtype=np.float32) for p in algo["polys"]]
                 if algo.get("polys") else None)
        self._alarm = BarrierAlarm(polys=polys)
        self._tracks: List[GapTracker] = []
        self.fg = _roi_mask(frame_shape, self._alarm.polys)

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
                                     self.algo["track"]["rer_purity"], self.algo["track"]["patch_size"])
                last_good["rer"] = rer
                return rer
            except Exception:
                return last_good["rer"]

        return confirm

    def step(self, frame: np.ndarray, now: float) -> List[TrackView]:
        """处理一帧：报警算法 -> 匹配轨道 -> 更新状态机；返回当前轨道视图。"""
        tcfg = self.algo["track"]
        gaps = self._alarm.step(frame)

        matched = {t: False for t in self._tracks}
        for g in gaps:
            hit = False
            for t in self._tracks:
                if _match(g.box, t.box, tcfg):
                    t.box = g.box
                    t.severity, t.kind = g.support, g.kind
                    t.last_seen = now
                    t.push_frame(frame)
                    t.update(GapCondition.DEFECTIVE, now, self._confirm(t))
                    matched[t] = True
                    hit = True
                    break
            if not hit:
                t = GapTracker(g.box, tcfg)
                t.severity, t.kind = g.support, g.kind
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
        return [TrackView(t.box, t.state, t.severity, t.kind, t._rer_cache[1])
                for t in self._tracks]


def _roi_mask(frame_shape: tuple, polys) -> np.ndarray:
    """ROI 并集掩膜（/mask.png 与叠加底图用；静态，每流计算一次）。"""
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
