"""Manual barrier ROIs, in annotation order left/far/right."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data/test/night_12000.json"
ANN_W, ANN_H = 1920, 1080

ROW_IDS = ["row_0_left_near", "row_1_far_center", "row_2_right_near"]


def load_manual_polys():
    """Return 3 polys in normalized (x/W, y/H) coords, order = left/far/right."""
    shapes = json.loads(ANN.read_text(encoding="utf-8"))["shapes"]
    polys = [np.array(s["points"], dtype=np.float32) / np.array([ANN_W, ANN_H], dtype=np.float32)
             for s in shapes]
    assert len(polys) == 3, f"expected 3 barrier polys, got {len(polys)}"
    return polys
