"""带绝对坐标网格的放大裁剪：人工勾 GT 多边形时读坐标用。

用法：python experiments/diagnostics/crop_grid.py <帧> x0 y0 x1 y1 [scale]
输出：results/_grid_<帧名>_<x0>_<y0>.png，网格线间隔 50px，标签为原图绝对坐标。
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_input

STEP, SCALE = 50, 2.0


def grid_crop(img, x0, y0, x1, y1, scale=SCALE):
    c = img[y0:y1, x0:x1].copy()
    c = cv.resize(c, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
    for x in range(x0 - x0 % STEP, x1, STEP):
        for y in range(y0 - y0 % STEP, y1, STEP):
            p = (int((x - x0) * scale), int((y - y0) * scale))
            cv.line(c, p, (p[0], p[1] + 8), (0, 255, 255), 1)
            cv.putText(c, f"x{x}y{y}", (p[0] + 2, p[1] + 14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv.LINE_AA)
    return c


def main():
    if len(sys.argv) < 6:
        sys.exit(f"用法: crop_grid.py <帧> x0 y0 x1 y1 [scale] (默认 scale={SCALE})")
    frame, x0, y0, x1, y1 = sys.argv[1], *map(int, sys.argv[2:6])
    scale = float(sys.argv[6]) if len(sys.argv) > 6 else SCALE
    stem = Path(frame).stem
    out = load_input(frame).shape and grid_crop(load_input(frame), x0, y0, x1, y1, scale)
    path = f"results/_grid_{stem}_{x0}_{y0}.png"
    cv.imwrite(path, out)
    print(path)


if __name__ == "__main__":
    main()
