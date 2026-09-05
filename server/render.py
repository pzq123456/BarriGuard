"""展示渲染：mask 绿色叠加 + 缺口状态框。"""
import cv2 as cv
import numpy as np

from .track import GapState

TINT, ALPHA = (80, 220, 80), 0.45
STATE_COLOR = {GapState.NORMAL: (120, 200, 120), GapState.SUSPECTED: (80, 160, 255),
               GapState.ALARM: (80, 80, 255)}


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    layer = frame.copy()
    layer[mask > 0] = TINT
    return cv.addWeighted(layer, ALPHA, frame, 1 - ALPHA, 0)


def draw_gaps(vis: np.ndarray, views) -> np.ndarray:
    """views：engine.Monitor.step 返回的 TrackView 序列（含 box/state/severity/kind/rer）。"""
    for v in views:
        x0, y0, x1, y1 = v.box
        color = STATE_COLOR[v.state]
        cv.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv.putText(vis, f"{v.kind}:{v.state.name}", (x0, max(20, y0 - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis
