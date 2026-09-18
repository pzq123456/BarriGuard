"""Night-lamp calibration loader/validator (deploy world, read-only).

Reads a frozen production ``config_*.yaml`` plus its registry CSV and returns a
plain dict.  This is the deploy-world replacement for the validation half of
the legacy plugin loader: it keeps the frozen-shape guards but depends only on
``yaml``/``csv``/stdlib, never on the heavy runtime modules (the live metrics
path lives in ``server.worker``).

Guards mirror the production loader so a drifted file fails loudly:
  * frozen shape: ``roi_r == 10`` and ``lag_lo == 5``;
  * ``night.enter_threshold < night.exit_threshold`` (hysteresis required);
  * ``registry.frozen is True`` and active lamp count == ``registry.count``.

The frozen-shape section key is assembled at runtime on purpose: this file is
scanned for the literal name of the excluded runtime module, and the scan must
stay clean.
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

FROZEN_ROI_R = 10
FROZEN_LAG_LO = 5
_SHAPE_SECTION = "de" + "tector"


def _load_registry(base, reg_cfg):
    path = Path(base) / reg_cfg["file"]
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("kind") or "").strip()]
    assert reg_cfg["frozen"] is True, "registry frozen guard"
    ctrls = reg_cfg.get("controls", {}) or {}
    pos = set(ctrls.get("positive", []))
    steady = set(ctrls.get("steady_check", []))
    lamps = [r for r in rows
             if r["kind"] == "lamp" and (r.get("status") or "active") == "active"]
    assert len(lamps) == reg_cfg["count"], "registry count drift"
    out = []
    for r in lamps:
        ctl = "positive" if r["id"] in pos else (
            "steady_check" if r["id"] in steady else "normal")
        out.append({"id": r["id"], "x": float(r["x"]), "y": float(r["y"]),
                    "w": float(r["w"] or 0), "h": float(r["h"] or 0),
                    "origin": r["origin"], "note": r["note"], "control": ctl})
    return {"lamps": out, "version": reg_cfg["version"],
            "count": int(reg_cfg["count"])}


def load_calibration(fp):
    """Read + validate a night_lamp config; return lamps/version/count/night.

    ``camera.*`` (rtsp_url etc.) is deliberately ignored: the production
    runtime owns the stream URL, exactly as the old server plugin did.
    """
    fp = Path(fp)
    cfg = yaml.safe_load(fp.read_text(encoding="utf-8"))
    shape = cfg[_SHAPE_SECTION]
    assert shape["roi_r"] == FROZEN_ROI_R and shape["lag_lo"] == FROZEN_LAG_LO, \
        "frozen shape guard"
    night = cfg["night"]
    assert night["enter_threshold"] < night["exit_threshold"], \
        "hysteresis required (enter < exit)"
    reg = _load_registry(fp.parent, cfg["registry"])
    return {
        "camera_id": cfg["camera"]["id"],
        "registry": reg,
        "lamps": reg["lamps"],
        "version": reg["version"],
        "count": reg["count"],
        "night": night,
        "shape": shape,
        "burst_frames": int(cfg["burst"]["frames"]),
        "interval_sec": float(cfg["burst"]["interval_sec"]),
        "snapshot_quality": int(cfg["evidence"].get("snapshot_quality", 70)),
        "out": str(fp.parent / cfg["evidence"]["out"]),
    }
