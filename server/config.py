"""配置加载：从 server/config.yaml 统一读取所有可调参数（多算法分块管理）。

- server / cameras / alert / log：服务级。
- algorithms.water_gap：水马缺口（分割 + 自主缺口 + 时序确认），映射到 SegmentCfg /
  GapCfg / TrackCfg。
- algorithms.night_lamp：夜间灯光（兄弟算法），仅解析原样保存，供后续接入。
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from water_barrier.research.segment import SegmentCfg
from water_barrier.research.detect import GapCfg

DEFAULT_PATH = Path(__file__).parent / "config.yaml"


@dataclass(frozen=True)
class ServerCfg:
    host: str
    port: int
    jpeg_quality: int
    preview_interval_s: float


@dataclass(frozen=True)
class CameraCfg:
    id: str
    name: str
    rtsp_url: str
    algorithm: str
    enabled: bool


@dataclass(frozen=True)
class AlertCfg:
    cooldown_s: float
    save_frame_overlay: bool


@dataclass(frozen=True)
class LogCfg:
    level: str


@dataclass(frozen=True)
class TrackCfg:
    decay_rate: float
    alarm_hold_s: float
    reconfirm_s: float
    rer_threshold: float
    track_stale_s: float
    match_iou: float
    match_center_ratio: float
    median_window: int
    intact_reset_s: float
    rer_purity: float
    patch_size: int


@dataclass(frozen=True)
class WaterGapAlgo:
    seg: SegmentCfg
    gap: GapCfg
    track: TrackCfg
    road_rois: tuple


@dataclass(frozen=True)
class Params:
    server: ServerCfg
    cameras: tuple
    water_gap: WaterGapAlgo
    night_lamp: dict
    alert: AlertCfg
    log: LogCfg

    def camera(self, cam_id: str) -> CameraCfg:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise KeyError(f"未找到相机: {cam_id}")

    def first_rtsp(self) -> str:
        for c in self.cameras:
            if c.enabled:
                return c.rtsp_url
        raise RuntimeError("没有启用任何相机")


def _water_gap(d: dict) -> WaterGapAlgo:
    c, cc, mo, rg, g, t = d["color"], d["cc"], d["morph"], d["region"], d["gap"], d["track"]
    seg = SegmentCfg(
        red_h=tuple(c["red_h"]),
        red_s_min=c["red_s_min"], red_a_min=c["red_a_min"],
        white_v_min=c["white_v_min"], white_s_max=c["white_s_max"],
        far_red_s_min=c["far_red_s_min"], far_red_a_min=c["far_red_a_min"],
        far_red_v_min=c["far_red_v_min"], far_white_v_min=c["far_white_v_min"],
        far_white_s_max=c["far_white_s_max"],
        cc_a_redish=cc["a_redish"], cc_v_whiteish=cc["v_whiteish"],
        cc_s_whiteish=cc["s_whiteish"], max_keep=cc["max_keep"],
        open_k=tuple(mo["open"]), close_k=tuple(mo["close"]), knit_h=tuple(mo["knit_h"]),
        knit_v=tuple(mo["knit_v"]), anchor_k=tuple(mo["anchor"]),
        min_cc_area=mo["min_cc_area"], far_min_cc_area=mo["far_min_cc_area"],
        roi=tuple(tuple(r) for r in rg["roi"]),
        far_band=tuple(tuple(r) for r in rg["far_band"]),
        exclude=tuple(tuple(r) for r in rg["exclude"]))
    gap = GapCfg(min_area=g["min_area"], min_gap_px=g["min_gap_px"],
                 max_gap_px=g["max_gap_px"], thr=g["thr"], min_len=g["min_len"],
                 end_margin=g["end_margin"], top_pieces=g["top_pieces"])
    track = TrackCfg(
        decay_rate=t["decay_rate"], alarm_hold_s=t["alarm_hold_s"],
        reconfirm_s=t["reconfirm_s"], rer_threshold=t["rer_threshold"],
        track_stale_s=t["track_stale_s"], match_iou=t["match_iou"],
        match_center_ratio=t["match_center_ratio"], median_window=t["median_window"],
        intact_reset_s=t["intact_reset_s"], rer_purity=t["rer_purity"],
        patch_size=t["patch_size"])
    return WaterGapAlgo(seg=seg, gap=gap, track=track,
                        road_rois=tuple(tuple(r) for r in rg["road_rois"]))


def load(path: str = None) -> Params:
    d = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    sv = d["server"]
    al = d["algorithms"]
    cams = tuple(CameraCfg(c["id"], c["name"], c["rtsp_url"], c["algorithm"], c["enabled"])
                 for c in d["cameras"])
    return Params(
        server=ServerCfg(sv["host"], sv["port"], sv["jpeg_quality"], sv["preview_interval_s"]),
        cameras=cams,
        water_gap=_water_gap(al["water_gap"]),
        night_lamp=al.get("night_lamp", {}),
        alert=AlertCfg(d["alert"]["cooldown_s"], d["alert"]["save_frame_overlay"]),
        log=LogCfg(d["log"]["level"]))
