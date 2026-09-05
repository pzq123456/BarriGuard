"""核心引擎：帧 -> 自主缺口检测 -> 时序确认。无固定 slot，无缺口坐标先验。

复用 water_barrier/research/ 的分割基座(segment.segment) 与自主检测(detect.detect_gaps)，
再用 server/track.py 的"衰减池 + 中值帧 RER 确认 + 迟滞状态机"做时序收敛，
把"真移除"与"人/车/遮挡"分开（单帧无法分离，靠时间+RER）。

全部阈值来自 server/config.yaml 的 WaterGapAlgo（segment/gap/track + 路面色框）。
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from .config import TrackCfg, WaterGapAlgo
from .track import GapCondition, GapState, GapTracker, RoadColorProfile, calc_patch_rer

from water_barrier.research.segment import segment
from water_barrier.research.detect import detect_gaps


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
    def __init__(self, frame_shape: tuple, algo: WaterGapAlgo) -> None:
        self.algo = algo
        self._gap = algo.gap
        self._road_rects = algo.road_rois
        self._tracks: List[GapTracker] = []
        self._next_id = 0
        self.fg = None

    def _confirm(self, track: GapTracker):
        """构建该缺口的 RER 确认闭包：中值帧重新分割 + 逐帧重建路色模型。"""
        last_good = {"rer": 0.0}

        def confirm(median_bgr):
            try:
                fg = segment(median_bgr, self.algo.seg)
                x0, y0, x1, y1 = track.box
                sm = np.zeros(fg.shape[:2], np.uint8)
                sm[max(0, y0):y1, max(0, x0):x1] = 255
                crops = [median_bgr[ry0:ry1, rx0:rx1] for rx0, ry0, rx1, ry1 in self._road_rects]
                crops = [c for c in crops if c.size > 0]
                if not crops:
                    return last_good["rer"]
                min_w = min(c.shape[1] for c in crops)
                if min_w <= 0:
                    return last_good["rer"]
                profile = RoadColorProfile(np.vstack([c[:, :min_w] for c in crops]))
                rer = calc_patch_rer(median_bgr, sm, fg, profile,
                                     self.algo.track.rer_purity, self.algo.track.patch_size)
                last_good["rer"] = rer
                return rer
            except Exception:
                return last_good["rer"]

        return confirm

    def step(self, frame: np.ndarray, now: float) -> List[TrackView]:
        """处理一帧：检测 -> 匹配轨道 -> 更新状态机；返回当前轨道视图。"""
        tcfg = self.algo.track
        fg = segment(frame, self.algo.seg)
        self.fg = fg
        gaps = detect_gaps(fg, self._gap)

        matched = {t: False for t in self._tracks}
        for g in gaps:
            hit = False
            for t in self._tracks:
                if _match((g.x0, g.y0, g.x1, g.y1), t.box, tcfg):
                    t.box = (g.x0, g.y0, g.x1, g.y1)
                    t.severity, t.kind = g.severity, g.kind
                    t.last_seen = now
                    t.push_frame(frame)
                    t.update(GapCondition.DEFECTIVE, now, self._confirm(t))
                    matched[t] = True
                    hit = True
                    break
            if not hit:
                t = GapTracker((g.x0, g.y0, g.x1, g.y1), self._next_id, tcfg)
                self._next_id += 1
                t.severity, t.kind = g.severity, g.kind
                t.last_seen = now
                t.push_frame(frame)
                t.update(GapCondition.DEFECTIVE, now, self._confirm(t))
                self._tracks.append(t)
                matched[t] = True

        for t in self._tracks:
            if not matched[t]:
                t.update(GapCondition.INTACT, now, self._confirm(t))

        self._tracks = [t for t in self._tracks
                        if now - t.last_seen < tcfg.track_stale_s
                        or t.state != GapState.NORMAL]
        return [TrackView(t.box, t.state, t.severity, t.kind, t._rer_cache[1])
                for t in self._tracks]


def _match(a: tuple, b: tuple, tcfg: TrackCfg) -> bool:
    if _iou(a, b) < tcfg.match_iou:
        return False
    gcx, gcy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    tcx, tcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    cd = ((gcx - tcx) ** 2 + (gcy - tcy) ** 2) ** 0.5
    return cd < tcfg.match_center_ratio * max(a[2] - a[0], b[2] - b[0])
