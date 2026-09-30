"""labelme 重标 -> roi_configs/<cam>.yaml（水马行 ROI 独立配置）。

* 行 id 已取消：pipeline 按顺序自动编为 row_0, row_1, ...；
* U（单个水马宽度，单位=轴向站）无法从标注得出，按新多边形轴向站数 M
  从旧行同比例换算（同机位同种水马，U 与 M 成正比），匹配行取形心最近的旧行；
* road_rois（路面颜色采样框）本次未重标，原样沿用旧文件。

用法：
    python deploy/tools/convert_roi.py --ann tmp/roi_relabel/1749_day.json
        --old deploy/app/water_barrier/configs/1749.yaml
        --out deploy/app/roi_configs/1749.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

TOOLS = Path(__file__).resolve().parent
APP = TOOLS.parent / "app"
REPO = TOOLS.parent.parent
sys.path.insert(0, str(APP))

import numpy as np  # noqa: E402

from water_barrier.signal import _row_axis  # noqa: E402

STEP_PX = 2.0


def _stations(poly_norm, w, h):
    pts = np.asarray(poly_norm, np.float32) * [w, h]
    ax = _row_axis(pts)
    length = float(ax["umax"] - ax["umin"])
    return max(1, int(np.ceil(length / STEP_PX)))


def convert(ann_path, old_path):
    ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    w, h = int(ann["imageWidth"]), int(ann["imageHeight"])
    old = yaml.safe_load(Path(old_path).read_text(encoding="utf-8")) or {}
    old_rows = old.get("rows", [])

    old_info = []
    for r in old_rows:
        pts = np.asarray(r["poly"], np.float32) * [w, h]
        old_info.append({"centroid": pts.mean(axis=0),
                         "U": float(r["U"]),
                         "M": _stations(r["poly"], w, h)})

    rows = []
    for i, s in enumerate(ann["shapes"]):
        assert s["shape_type"] == "polygon", s["shape_type"]
        pts = np.asarray(s["points"], np.float32)
        poly = (pts / [w, h]).round(6).tolist()
        m = _stations(poly, w, h)
        if old_info:
            dists = [float(np.linalg.norm(pts.mean(axis=0) - o["centroid"]))
                     for o in old_info]
            o = old_info[int(np.argmin(dists))]
            u = round(o["U"] * m / o["M"], 1)
            basis = "U_old=%.1f M_old=%d" % (o["U"], o["M"])
        else:
            u, basis = None, "无旧行可换算，需手填 U"
        rows.append({"poly": poly, "U": u, "M_new": m, "U_basis": basis,
                     "label": s.get("label")})
    return {"rows": rows, "road_rois": old.get("road_rois", []),
            "image": (f"{w}x{h} <- {Path(ann_path).name}")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    d = convert(a.ann, a.old)
    missing = [i for i, r in enumerate(d["rows"]) if r["U"] is None]
    if missing:
        raise SystemExit(f"行 {missing} 无 U 可换算，请先手填")

    lines = ["# ROI 数据资产：水马行多边形（用户 labelme 重标，行 id 已取消，",
             "# pipeline 按顺序自动编为 row_0, row_1, ...）。",
             f"# 来源：{d['image']}；U 按轴向站数 M 从旧行同比例换算；",
             "# road_rois（路面颜色采样框）本次未重标，沿用旧文件。"]
    body = {"rows": [{k: r[k] for k in ("poly", "U")} for r in d["rows"]],
            "road_rois": d["road_rois"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n"
                           + yaml.safe_dump(body, allow_unicode=True,
                                            sort_keys=False),
                           encoding="utf-8")
    for i, r in enumerate(d["rows"]):
        print(f"row_{i}: M={r['M_new']} U={r['U']} ({r['U_basis']})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
