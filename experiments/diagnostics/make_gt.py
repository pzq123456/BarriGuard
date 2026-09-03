"""渲染 GT：多边形 → data/gt/<帧名>_gt.png，并输出叠加目检图。

用法：python experiments/diagnostics/make_gt.py <帧名不含扩展名>
"""
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "gt"))
sys.path.insert(0, str(ROOT / "experiments"))
from common import load_input  # noqa: E402
from gt_polys import GT  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GT_DIR = ROOT / "data" / "gt"


def build_gt(img, spec):
    gt = np.zeros(img.shape[:2], np.uint8)
    for poly in spec["polys"]:
        cv.fillPoly(gt, [np.array(poly, np.int32)], 255)
    for x0, y0, x1, y1 in spec["holes"]:
        gt[y0:y1, x0:x1] = 0
    return gt


def overlay(img, gt):
    vis = img.copy()
    layer = vis.copy()
    layer[gt > 0] = (255, 0, 255)  # GT 品红
    vis = cv.addWeighted(layer, 0.45, vis, 0.55, 0)
    for x0, y0, x1, y1 in spec_holes:
        cv.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return vis


def main():
    stem = sys.argv[1]
    spec = GT[stem]
    global spec_holes
    spec_holes = spec["holes"]
    img = load_input(stem + ".png")
    gt = build_gt(img, spec)
    out = GT_DIR / f"{stem}_gt.png"
    cv.imwrite(str(out), gt)
    cv.imwrite(str(ROOT / "results" / f"_gtchk_{stem}.png"), overlay(img, gt))
    print(f"{out}  px={np.count_nonzero(gt)}")
    print(json.dumps({"polys": len(spec["polys"]), "holes": len(spec["holes"])}))


if __name__ == "__main__":
    main()
