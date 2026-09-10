"""water_gap 算法主体 (实现 server.Algorithm 协议): 帧 -> 缺口事件 + 标注 + 调试图层。

分层: signal 是单帧候选纯函数；track 是 support 累计 + "衰减池 + 路露率 RER
确认 + 迟滞状态机"，把"真移除"与"人/车长期遮挡"分开 (单帧无法分离，靠时间+RER)；
本模块只做装配（标定加载 + 轨道匹配 + server 协议）。

RER 的"水马前景"改用红/白颜色掩膜 (轴向采样不输出分割图)，
语义不变: 缺口框内非水马像素中真实路面 patch 占比。

标定文件结构见 water_barrier/configs/1749.yaml：rows（id/poly/U）、road_rois、
track、detect（8 键全顶层）。只留近行；配置行即上报行。

对外只出三样东西 (见 server/algo.py):
  events  非 NORMAL 轨道 (alarm/suspected, 含 box/severity/rer/row_id)
  annots  全部轨道的框标注 (含 NORMAL, 供叠加渲染)
  debug   {"roi_mask": ROI 并集掩膜} (按需加层, server 原样 serving)
"""
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

from server.algo import AlgoResult, Annotation, Event

from . import track as T
from .signal import red_m, white_m

# 时序确认参数（12 个，逐帧 RER 确认 + 状态机用，见 track.py）。
TRACK_KEYS = ("decay_rate", "alarm_hold_s", "reconfirm_s", "rer_threshold",
              "track_stale_s", "match_iou", "match_center_ratio",
              "median_window", "intact_reset_s", "rer_purity", "patch_size",
              "sunglare_road_white")

# 检测参数（8 个，全顶层：前 4 support 累计用，后 4 signal.detect 成核用）。
DETECT_KEYS = ("trim", "conf_th", "support_th", "min_box_width",
               "floor_max", "core_exit", "min_width", "core_min_width")

# 状态机状态 -> server 通用配色。
LEVEL = {T.ALARM: "alarm", "SUSPECTED": "suspected", T.NORMAL: "info"}


def _need(d, keys, ctx):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            raise RuntimeError(f"标定缺字段 {ctx}.{k}")


def load_calibration(fp):
    """读一份标定文件并校验（只留近行；配置行即上报行，无 report 开关）。"""
    fp = Path(fp)
    d = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
    rows = d.get("rows", [])
    if not rows:
        raise RuntimeError(f"标定缺rows: {fp}")
    ids = [r.get("id") for r in rows]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise RuntimeError(f"rows.id 非法/重复: {fp}")
    for r in rows:
        if not r.get("poly"):
            raise RuntimeError(f"行缺poly: {fp}.{r['id']}")
        if r.get("U") is None:
            raise RuntimeError(f"行缺U标定: {fp}.{r['id']}")
    _need(d, ("road_rois", "track"), str(fp))
    _need(d["track"], TRACK_KEYS, str(fp))
    d.setdefault("detect", {})
    for k in d["detect"]:
        if k not in DETECT_KEYS:
            raise ValueError(f"未知detect键: {fp}.detect.{k}")
    return d


def _iou(a, b):
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
    """water_gap per_frame 算法, 状态为实例属性 (多流各持一份)。

    server 协议要求 class（name/cadence/step/reset）；内部状态均为 plain 数据。
    """

    name = "water_gap"
    cadence = "per_frame"

    def __init__(self, frame_shape, calib, camera_id=""):
        for k in calib.get("detect", {}):
            if k not in DETECT_KEYS:
                raise ValueError(f"未知detect键: {k}")
        self._cam = camera_id
        self._tcfg = calib["track"]
        self._road_rects = calib["road_rois"]
        rows = calib["rows"]
        det = calib.get("detect", {})
        polys = [np.asarray(r["poly"], dtype=np.float32) for r in rows]
        self._sup = T.new_support_state(
            polys=polys, row_ids=[r["id"] for r in rows],
            units={r["id"]: r["U"] for r in rows},
            trim=det.get("trim", T.DEFAULT_TRIM),
            conf_th=det.get("conf_th", T.DEFAULT_CONF_TH),
            support_th=det.get("support_th", T.DEFAULT_SUPPORT_TH),
            min_box_width=det.get("min_box_width", T.DEFAULT_MIN_BOX_WIDTH),
            floor_max=det.get("floor_max", T.DEFAULT_FLOOR_MAX),
            core_exit=det.get("core_exit", T.DEFAULT_CORE_EXIT),
            min_width=det.get("min_width", T.DEFAULT_MIN_WIDTH),
            core_min_width=det.get("core_min_width", T.DEFAULT_CORE_MIN_WIDTH))
        self._tracks = []
        self.fg = _roi_mask(frame_shape, polys)

    def reset(self):
        self._tracks = []
        T.reset_support(self._sup)

    def _calc_rer(self, median_bgr, box):
        """框内路露率（失败回 0.0）。"""
        try:
            x0, y0, x1, y1 = box
            sm = np.zeros(median_bgr.shape[:2], np.uint8)
            sm[max(0, y0):y1, max(0, x0):x1] = 255
            crops = [median_bgr[ry0:ry1, rx0:rx1] for rx0, ry0, rx1, ry1 in self._road_rects]
            crops = [c for c in crops if c.size > 0]
            if not crops:
                return 0.0
            min_w = min(c.shape[1] for c in crops)
            if min_w <= 0:
                return 0.0
            profile = T.fit_road_profile(np.vstack([c[:, :min_w] for c in crops]))
            fg = np.where(red_m(median_bgr) | white_m(median_bgr),
                          255, 0).astype(np.uint8)
            return T.calc_patch_rer(median_bgr, sm, fg, profile,
                                    self._tcfg["rer_purity"], self._tcfg["patch_size"])
        except Exception:
            return 0.0

    def step(self, frame, now):
        """处理一帧：报警算法 -> 匹配轨道 -> 更新状态机。

        SUNGLARE 只定状态（经 debug.frame_status 透出，UNKNOWN，不是 NORMAL），
        不冻结检测：眩光帧的真缺口仍可能经 RER 确认，误报仍由 RER 挡。
        """
        tcfg = self._tcfg
        road_white = _road_white_frac(frame, self._road_rects)
        status = "SUNGLARE" if (self._road_rects
                                and road_white > tcfg["sunglare_road_white"]) else "OK"
        gaps = T.step_support(self._sup, frame)

        matched = {id(t): False for t in self._tracks}
        for row_id, _sta, box, support in gaps:
            hit = None
            for t in self._tracks:
                if _match(box, t["box"], tcfg):
                    hit = t
                    break
            if hit is None:
                hit = T.new_track(box, tcfg)
                self._tracks.append(hit)
                matched[id(hit)] = False
            hit["box"] = box
            hit["severity"], hit["row_id"] = support, row_id
            hit["last_seen"] = now
            T.track_push_frame(hit, frame)
            T.track_update(hit, True, now,
                           lambda med, _b=box: self._calc_rer(med, _b))
            matched[id(hit)] = True

        for t in self._tracks:
            if not matched.get(id(t), False):
                T.track_update(t, False, now,
                               lambda med, _b=t["box"]: self._calc_rer(med, _b))

        self._tracks = [t for t in self._tracks
                        if now - t["last_seen"] < tcfg["track_stale_s"]
                        or t["state"] != T.NORMAL]
        annots = [Annotation("box", t["box"], t["state"], LEVEL[t["state"]],
                             {"severity": t["severity"],
                              "rer": round(float(t["rer"]), 3),
                              "row_id": t["row_id"]})
                  for t in self._tracks]
        events = [Event(self._cam, "water_gap", now,
                        "alarm" if t["state"] == T.ALARM else "suspected",
                        {"box": t["box"], "severity": t["severity"],
                         "rer": round(float(t["rer"]), 3), "row_id": t["row_id"]})
                  for t in self._tracks if t["state"] != T.NORMAL]
        return AlgoResult(events, annots, {"roi_mask": self.fg, "frame_status": status,
                                           "road_white": round(road_white, 3)})


def _road_white_frac(frame, rects):
    """路面取色框的 white 占比中位数（日光门控用；无框时返回 0 即不过门）。"""
    vals = []
    for rx0, ry0, rx1, ry1 in rects:
        crop = frame[max(0, ry0):ry1, max(0, rx0):rx1]
        if crop.size > 0:
            vals.append(float((white_m(crop)).mean()))
    return float(np.median(vals)) if vals else 0.0


def _roi_mask(frame_shape, polys):
    """ROI 并集掩膜（debug 图层用；静态，每流计算一次）。"""
    h, w = frame_shape[:2]
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv.fillPoly(m, [(np.asarray(p, dtype=np.float32) * [w, h]).astype(int)], 255)
    return m


def _match(a, b, tcfg):
    if _iou(a, b) < tcfg["match_iou"]:
        return False
    gcx, gcy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    tcx, tcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    cd = ((gcx - tcx) ** 2 + (gcy - tcy) ** 2) ** 0.5
    return cd < tcfg["match_center_ratio"] * max(a[2] - a[0], b[2] - b[0])
