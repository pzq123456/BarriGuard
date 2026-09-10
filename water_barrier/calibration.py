"""water_gap 标定加载与校验 (领域归算法包所有, server 只管分发文件路径)。

标定文件结构见 water_barrier/configs/1749.yaml：rows（id/poly/U/report）、
road_rois、track、detect（可选覆盖，键见 pipeline.DETECT_KEYS）。
"""
from pathlib import Path

import yaml

TRACK_KEYS = ("decay_rate", "alarm_hold_s", "reconfirm_s", "rer_threshold",
              "track_stale_s", "match_iou", "match_center_ratio",
              "median_window", "intact_reset_s", "rer_purity", "patch_size",
              "sunglare_road_white")


def _need(d, keys, ctx):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            raise RuntimeError(f"标定缺字段 {ctx}.{k}")


def load_calibration(fp):
    """读一份标定文件并校验。"""
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
        r.setdefault("report", False)
    _need(d, ("road_rois", "track"), str(fp))
    _need(d["track"], TRACK_KEYS, str(fp))
    d.setdefault("detect", {})
    return d
